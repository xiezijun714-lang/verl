# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import re
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


@dataclass
class TrajectoryOutput:
    """SUPO: 单条子轨迹的输出数据"""
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float]
    is_final: bool = False


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        self.routed_experts = None

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}

        # ========== SUPO (Summarization augmented Policy Optimization) 相关字段 ==========
        self.original_messages: list[dict] = []  # 保存原始messages用于摘要后重建prompt
        self.original_prompt_ids: list[int] = []  # 保存原始prompt s_1
        self.summary_count: int = 0  # 已执行的摘要/压缩轮次 I
        self.trajectory_outputs: list[TrajectoryOutput] = []  # 累积的子轨迹输出
        self.is_summarizing: bool = False  # 当前是否在生成摘要/选择（v_sum ∈ s_t）
        self.overlong: bool = False  # 是否因超长被终止
        # 当前trajectory累积的response（用于保存trajectory）
        self.current_traj_prompt_ids: list[int] = []
        self.accumulated_response_ids: list[int] = []
        self.accumulated_response_mask: list[int] = []
        self.accumulated_logprobs: list[float] = []

        # ========== ECHO: turn-level history for context reconstruction ==========
        self.turn_history: list[dict] = []  # [{"query": str, "finding": str, "tool_response_text": str}]


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize tools from config file
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        # Initialize interactions from config file
        self.interaction_config_file = self.rollout_config.multi_turn.interaction_config_path
        if self.interaction_config_file:
            self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(
                self.interaction_config_file
            )

        # ========== SUPO (Summarization augmented Policy Optimization) 配置 ==========
        self.enable_summarization = getattr(self.rollout_config.multi_turn, 'enable_summarization', False)
        self.max_summary_rounds = getattr(self.rollout_config.multi_turn, 'max_summary_rounds', 2)
        self.working_context_length = getattr(self.rollout_config.multi_turn, 'working_context_length', 8192)
        # context compression method: "summary" (SUPO) or "echo_e2e" (ECHO: actor selects turns)
        self.context_compression_method = getattr(self.rollout_config.multi_turn, 'context_compression_method', 'summary')
        self.summary_instruction = getattr(
            self.rollout_config.multi_turn, 
            'summary_instruction', 
            "System:\nYou are a helpful agent interacting with a function calling environment to solve user's problem. The interaction history is now too long. Please summarize the interaction history.\n• Remember to keep the important information in the history to ensure that you can continue solving the problem.\n• Do not call any function in this turn.\nNow generate the summary, and put your summary inside tag <summary></summary>.\n"
        )
        self.selection_instruction = getattr(
            self.rollout_config.multi_turn,
            'selection_instruction',
            "[CONTEXT COMPRESSION] Your context is full. Previous interaction turns:\n{turn_list}\n\nSelect the most relevant turns to keep for solving the task.\nPut your selection inside <selection></selection> tags, one turn per line. Format: turn_N: reason\nExample:\nturn_0: contains the initial query constraints\nturn_3: has key API response data\nDo NOT output answers to the original task. ONLY output the selection tag."
        )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput | list[AgentLoopOutput]:
        messages = list(kwargs["raw_prompt"])

        # extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # ========== SUPO: 多条trajectory输出 ==========
        if self.enable_summarization:
            # 保存最终trajectory（如果不是overlong）
            if not agent_data.overlong and agent_data.accumulated_response_ids:
                self._save_final_trajectory(agent_data)
            
            # 生成多条AgentLoopOutput
            outputs = []
            multi_modal_data_dict = {}
            if agent_data.image_data is not None:
                multi_modal_data_dict["images"] = agent_data.image_data
            if agent_data.video_data is not None:
                multi_modal_data_dict["videos"] = agent_data.video_data
            
            rollout_id = f"{request_id}_rollout"
            uid = kwargs.get("uid", request_id)
            is_padding = kwargs.get("is_padding", False)  # Get is_padding marker for SUPO unpad
            reward_model = kwargs.get("reward_model", {})  # Get reward_model for SUPO (contains ground_truth)
            data_source = kwargs.get("data_source", "unknown")  # Get data_source for metrics
            
            for i, traj in enumerate(agent_data.trajectory_outputs):
                # Truncate prompt and response to fit within max_model_len
                # prompt_ids: take last prompt_length tokens if too long (left truncate)
                # response_ids: take first response_length tokens if too long (right truncate)
                truncated_prompt_ids = traj.prompt_ids[-self.prompt_length:] if len(traj.prompt_ids) > self.prompt_length else traj.prompt_ids
                truncated_response_ids = traj.response_ids[:self.response_length]
                truncated_response_mask = traj.response_mask[:self.response_length]
                truncated_logprobs = traj.response_logprobs[:self.response_length] if traj.response_logprobs else None
                
                output = AgentLoopOutput(
                    prompt_ids=truncated_prompt_ids,
                    response_ids=truncated_response_ids,
                    response_mask=truncated_response_mask,
                    multi_modal_data=multi_modal_data_dict if i == len(agent_data.trajectory_outputs) - 1 else {},
                    response_logprobs=truncated_logprobs,
                    num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
                    metrics=agent_data.metrics,
                    routed_experts=agent_data.routed_experts if i == len(agent_data.trajectory_outputs) - 1 else None,
                    extra_fields={
                        "traj_idx": i,
                        "is_final": traj.is_final,
                        "overlong": agent_data.overlong,
                        "rollout_id": rollout_id,
                        "uid": uid,
                        "is_padding": is_padding,  # Pass through for SUPO unpad
                        "reward_model": reward_model,  # Pass through for reward calculation
                        "data_source": data_source,  # Pass through for metrics
                        "turn_scores": agent_data.turn_scores,
                        "tool_rewards": agent_data.tool_rewards,
                    },
                )
                outputs.append(output)
            
            logger.debug(
                f"SUPO: Rollout completed. Generated {len(outputs)} trajectories. "
                f"overlong={agent_data.overlong}, summary_count={agent_data.summary_count}"
            )
            
            
            return outputs

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=agent_data.routed_experts,
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        # ECHO: 在 system prompt 中注入 sum_last_turn 规则（类似 tool_call 格式说明，只注入一次）
        if self.enable_summarization and self.context_compression_method == "echo_e2e":
            sum_instruction = (
                "\n\n# Response Format\n"
                "After receiving each tool response, you MUST include a brief summary of the key finding "
                "from that tool response using <sum_last_turn></sum_last_turn> tags, "
                "before making your next tool call."
            )
            for msg in agent_data.messages:
                if msg.get("role") == "system":
                    msg["content"] = msg["content"] + sum_instruction
                    break

        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=self.tool_schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        
        # SUPO: 保存原始prompt和messages用于摘要后重置
        if self.enable_summarization:
            agent_data.original_prompt_ids = list(prompt_ids)
            agent_data.original_messages = list(agent_data.messages)  # 保存原始messages
            # current_traj_prompt_ids 只保存原始 prompt（system + user + assistant waiting）
            # 所有历史内容将放到 response 中，确保有梯度
            agent_data.current_traj_prompt_ids = list(prompt_ids)
        
        return AgentState.GENERATING


    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""

        with simple_timer("generate_sequences", agent_data.metrics):
            generate_prompt_ids = agent_data.prompt_ids

            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=generate_prompt_ids,  # 使用拼接后的 prompt
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )

        # ECHO selection 阶段：生成后在 </selection> 处截断，防止模型继续生成重复内容
        # （sglang 配置了 skip_tokenizer_init=True，server 端无法用字符串 stop；
        #  而 </selection> 的最后 token 是 '>', 用 stop_token_ids 会误停于任何 '>' 字符，
        #  因此只能生成后做后处理截断）
        # ECHO selection 阶段：生成后在 </selection> 处截断
        if (self.enable_summarization and agent_data.is_summarizing
                and self.context_compression_method == "echo_e2e"):
            
            # 先整体 Decode 一次，看是否需要截断（性能优化）
            decoded_output = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
            
            if "</selection>" in decoded_output:
                original_token_len = len(output.token_ids)
                end_idx = original_token_len
                
                # 逐个 Token 向后 Decode，找到刚好闭合 "</selection>" 的那个 Token 位置
                # 虽然是个 for 循环，但 token 数组长度通常在几百以内，Decode 极快，对整体耗时可忽略不计
                for i in range(1, original_token_len + 1):
                    prefix_text = self.tokenizer.decode(output.token_ids[:i], skip_special_tokens=True)
                    if "</selection>" in prefix_text:
                        end_idx = i
                        break
                
                # 如果找到了截断点，并且截断点小于原长度（说明后面有废话）
                if end_idx < original_token_len:
                    # 直接对原数组切片，完美保留 log_probs 对齐，无需 re-encode
                    output.token_ids = output.token_ids[:end_idx]
                    if output.log_probs:
                        output.log_probs = output.log_probs[:end_idx]
                    
                    logger.warning(
                        f"[ECHO-DBG trunc-success] req={agent_data.request_id} "
                        f"Tokens truncated: {original_token_len} -> {end_idx}"
                    )
            else:
                logger.warning(
                    f"[ECHO-DBG trunc-miss] req={agent_data.request_id} "
                    f"No </selection> tag found in the output!"
                )
        
        # SUPO: 摘要生成不增加 assistant_turns
        if not (self.enable_summarization and agent_data.is_summarizing):
            agent_data.assistant_turns += 1

        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        # ECHO selection 阶段：诊断 log
        if (self.enable_summarization and agent_data.is_summarizing
                and self.context_compression_method == "echo_e2e"):
            decoded = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            n_sum_open = decoded.count("<sum_last_turn>")
            n_sum_close = decoded.count("</sum_last_turn>")
            n_sel_open = decoded.count("<selection>")
            n_sel_close = decoded.count("</selection>")
            logger.warning(
                f"[ECHO-DBG gen] req={agent_data.request_id} sum_cnt={agent_data.summary_count} "
                f"resp_len={len(agent_data.response_ids)} "
                f"sum_open={n_sum_open} sum_close={n_sum_close} sel_open={n_sel_open} sel_close={n_sel_close}"
            )
            if n_sel_open > 1 or n_sum_open > 1:
                # 找到第二个 <selection> 在文本中的字符偏移，打印周围上下文
                parts = decoded.split("<selection>")
                logger.warning(
                    f"[ECHO-DBG gen] REPEAT DETECTED in single generation. "
                    f"first_sel_ctx={parts[1][:150]!r} ... second_sel_ctx={parts[2][:150]!r}"
                    if len(parts) > 2 else
                    f"[ECHO-DBG gen] multi sum_last_turn but single selection"
                )

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # 处理上下文压缩生成阶段（summary 或 turn selection）
        if self.enable_summarization and agent_data.is_summarizing:
            # 追加压缩指令+输出到当前 trajectory
            self._append_summary_to_current_trajectory(agent_data, output.log_probs or [])

            # 解码输出文本
            full_output = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )

            compression_ok = True
            # 预初始化 ECHO 专用变量，保证即便 SUPO 分支或无历史分支，后续引用也不会 UnboundLocalError
            selected_indices: list = []
            pending = None
            if self.context_compression_method == "echo_e2e":
                # ECHO: 提取 pending_turn 的 finding（（<sum_last_turn> 格式）
                pending = getattr(agent_data, 'pending_turn', None)
                if pending:
                    finding = self._extract_turn_finding(full_output)
                    if finding:
                        pending["finding"] = finding

                if agent_data.turn_history:
                    # 有历史轮：解析 selection + 构建新 prompt
                    parsed = self._parse_selection_indices(full_output, len(agent_data.turn_history))
                    if parsed is None:
                        # selection 解析失败，fallback：保留全部历史轮
                        logger.warning(f"ECHO: Selection parse failed, falling back to keep all {len(agent_data.turn_history)} turns.")
                        parsed = [{"index": i, "score": 0.5} for i in range(len(agent_data.turn_history))]
                    selected_indices = parsed
                    new_messages = self._build_echo_prompt(agent_data, selected_indices, pending)
                else:
                    # 首轮超长：无历史轮可选，只做了 summary，直接用 pending_turn 构建
                    new_messages = self._build_echo_prompt(agent_data, [], pending)
            else:
                # SUPO: 用 summary 文本重构 prompt
                summary_text = self._summary_exclude_think(full_output)
                if len(summary_text.strip()) < 10:
                    logger.warning(f"SUPO: Summary too short ({repr(summary_text[:50])}), compression failed.")
                    compression_ok = False
                else:
                    new_messages = [
                        *agent_data.original_messages,
                        {"role": "assistant", "content": f"<summary>{summary_text}</summary>"},
                        {"role": "user", "content": "The above is a summary of your previous interaction history. Please continue solving the problem based on this context. You may call tools as needed."},
                    ]

            if not compression_ok:
                # 压缩输出质量太差，终止
                agent_data.is_summarizing = False
                agent_data.overlong = True
                logger.warning("Context compression output unusable. Terminating.")
                return AgentState.TERMINATED

            new_prompt_ids = await self.apply_chat_template(new_messages, tools=self.tool_schemas)

            # ECHO: 压缩后检查 prompt 是否仍然过长
            # 如果 rebuild 后 prompt 占比 > 80% working_context_length，后续 tool_response 几乎必然再次 overflow
            # 直接终止，避免无效的压缩死循环（sum_cnt 0→1→2→...→max）
            if self.context_compression_method == "echo_e2e":
                prompt_ratio = len(new_prompt_ids) / self.working_context_length if self.working_context_length > 0 else 0
                if prompt_ratio > 0.8:
                    logger.warning(
                        f"[ECHO] Post-compression prompt still too long: {len(new_prompt_ids)} tokens "
                        f"({prompt_ratio:.1%} of working_context_length={self.working_context_length}). "
                        f"Terminating to avoid compression loop."
                    )
                    agent_data.is_summarizing = False
                    agent_data.overlong = True
                    return AgentState.TERMINATED

            agent_data.prompt_ids = new_prompt_ids
            agent_data.messages = new_messages
            
            agent_data.current_traj_prompt_ids = list(agent_data.prompt_ids)
            # 清空历史累积
            agent_data.accumulated_response_ids = []
            agent_data.accumulated_response_mask = []
            agent_data.accumulated_logprobs = []
            
            # 重置标记
            agent_data.is_summarizing = False
            agent_data.summary_count += 1
            # ECHO: 重建后按 selection 更新 turn_history（而不是清空），
            # 保持与新 prompt 中 [turn_X Key Finding] 的一致性；pending_turn 若有 finding 则并入历史
            if self.context_compression_method == "echo_e2e":
                new_turn_history = [
                    agent_data.turn_history[item["index"]]
                    for item in selected_indices
                    if item["index"] < len(agent_data.turn_history)
                ]
                if pending and pending.get("finding"):
                    new_turn_history.append(pending)
                agent_data.turn_history = new_turn_history
                agent_data.pending_turn = None

            return AgentState.GENERATING

        if self.enable_summarization:
            agent_data.accumulated_response_ids.extend(agent_data.response_ids)
            agent_data.accumulated_response_mask.extend([1] * len(agent_data.response_ids))
            agent_data.accumulated_logprobs.extend(output.log_probs or [0.0] * len(agent_data.response_ids))

        # 检查终止条件（保持原逻辑）
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # ... 其余检查工具调用逻辑保持不变 ...
        is_last_turn = self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns
        if is_last_turn:
            agent_data.tool_calls = []
            return AgentState.TERMINATED

        # Extract tool calls
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        # 单轮生成被 max_tokens 截断且没有合法 tool_call
        if output.stop_reason == "length" and not agent_data.tool_calls:
            if self.enable_summarization and not agent_data.is_summarizing \
                    and agent_data.summary_count < self.max_summary_rounds \
                    and len(agent_data.accumulated_response_ids) > 0:
                # 有历史轮次：回滚当前废 think，保存子轨迹，压缩后继续
                logger.warning(
                    f"Single-turn think truncated (length={len(agent_data.response_ids)}), "
                    f"but has prior turns. Rolling back and compressing."
                )
                # ECHO 2b: 回滚前检查截断 think 里是否已有 sum_last_turn
                if self.context_compression_method == "echo_e2e":
                    truncated_text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
                    finding = self._extract_turn_finding(truncated_text)
                    pending = getattr(agent_data, 'pending_turn', None)
                    if finding and pending:
                        pending["finding"] = finding
                        logger.info(f"ECHO 2b: Extracted finding from truncated think, skip sum in selection.")

                # 回滚当前轮的 a_t（被截断的 think）
                a_t_len = len(agent_data.response_ids)
                if a_t_len > 0:
                    agent_data.accumulated_response_ids = agent_data.accumulated_response_ids[:-a_t_len]
                    agent_data.accumulated_response_mask = agent_data.accumulated_response_mask[:-a_t_len]
                    agent_data.accumulated_logprobs = agent_data.accumulated_logprobs[:-a_t_len]
                self._save_current_trajectory_without_current_turn(agent_data)
                if self.context_compression_method == "echo_e2e":
                    return await self._trigger_echo_selection(agent_data)
                else:
                    return await self._trigger_summary(agent_data)
            else:
                # 第一轮就超长 / 压缩次数用完 / 已在压缩中 → 直接终止
                agent_data.overlong = True
                logger.warning(
                    f"Single-turn think truncated (length={len(agent_data.response_ids)}), "
                    f"no prior turns or compression exhausted. Terminating."
                )
                return AgentState.TERMINATED

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)
        # Determine next state
        if agent_data.tool_calls:
            # ECHO: pending_turn 机制 —— 上一轮 pending 获取 finding 后 append，新轮暂存为 pending
            if self.enable_summarization and self.context_compression_method == "echo_e2e":
                # 上一轮 pending_turn 获取 finding 后正式 append 到 turn_history
                pending = getattr(agent_data, 'pending_turn', None)
                if pending:
                    if not pending.get("finding"):
                        full_text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
                        pending["finding"] = self._extract_turn_finding(full_text)
                    agent_data.turn_history.append(pending)
                    agent_data.pending_turn = None

                # 提取纯 query，暂存为 pending_turn（不 append 到 turn_history）
                query_str = ""
                try:
                    for tc in agent_data.tool_calls:
                        args = json.loads(tc.arguments)
                        if isinstance(args, dict) and "query" in args:
                            query_str = args["query"]
                            break
                except Exception:
                    query_str = ", ".join(f"{tc.name}({tc.arguments})" for tc in agent_data.tool_calls)[:100]
                agent_data.pending_turn = {
                    "query": query_str,
                    "finding": "",  # selection 或下一轮 think 时填充
                    "tool_response_text": "",  # tool_response 返回后填充
                }
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED


    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_response, tool_reward, _ in responses:
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                 # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        # ECHO: 补充当前 turn 的 tool_response 文本（写入 pending_turn），并把 sum_last_turn hint
        # 作为独立 user message 一起参与本次模板编码，避免后续 ID 拼接破坏 chat template 边界。
        echo_inject_hint = (
            self.enable_summarization
            and self.context_compression_method == "echo_e2e"
            and hasattr(agent_data, 'pending_turn')
            and agent_data.pending_turn
        )
        if echo_inject_hint:
            tool_resp_text = " | ".join(msg.get("content", "") for msg in add_messages)
            agent_data.pending_turn["tool_response_text"] = tool_resp_text
            hint_msg = {
                "role": "user",
                "content": "Summarize the key finding from the last tool response in <sum_last_turn></sum_last_turn>, then continue.",
            }
            encode_messages = add_messages + [hint_msg]
        else:
            encode_messages = add_messages

        agent_data.messages.extend(encode_messages)

        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            tool_response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
            if echo_inject_hint:
                hint_ids = await self.apply_chat_template([hint_msg], remove_system_prompt=True)
                tool_response_ids = tool_response_ids + hint_ids
        else:
            # Note that we have to pass None to the images and videos if there are no new images / videos
            # to stay compatible with downstream image processing logic!
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            tool_response_ids = await self.apply_chat_template(
                encode_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

        # ========== 上下文超限检测 ==========
        if self.enable_summarization:
            # 总长度 = prompt + 累积response + 当前a_t + 工具结果o_t
            current_total_length = (
                len(agent_data.prompt_ids) +           # 基础 prompt (s_1 + 可能的前一个summary)
                len(agent_data.accumulated_response_ids) +  # 历史累积的 response
                len(tool_response_ids)                 # 工具返回 o_t
            )
            
            if current_total_length >= self.working_context_length:
                logger.debug(
                    f"Context overflow detected. "
                    f"total={current_total_length} >= L={self.working_context_length}. "
                    f"method={self.context_compression_method}"
                )
                
                if agent_data.summary_count < self.max_summary_rounds:
                    if self.context_compression_method == "echo_e2e":
                        # ECHO: 连续 overflow 且 turn_history <= 1 → 压缩无法缩短，直接终止
                        n_hist = len(getattr(agent_data, 'turn_history', []))
                        pending_finding = bool(
                            hasattr(agent_data, 'pending_turn') and agent_data.pending_turn
                            and agent_data.pending_turn.get("finding")
                        )
                        if agent_data.summary_count > 0 and n_hist <= 1 and not pending_finding:
                            agent_data.overlong = True
                            logger.warning(
                                f"[ECHO] Consecutive overflow with turn_history_len={n_hist}, "
                                f"sum_cnt={agent_data.summary_count}. Compression cannot help. Terminating."
                            )
                            self._save_current_trajectory_without_current_turn(agent_data)
                            return AgentState.TERMINATED

                        # ECHO: tool_response 因 overflow 被截断后无意义，回滚 a_t 并丢弃 pending_turn，
                        # 保证 trajectory 和新 prompt 的一致性（不遗留"发了 search 但没结果"的幽灵状态）
                        a_t_len = len(agent_data.response_ids)
                        if a_t_len > 0:
                            agent_data.accumulated_response_ids = agent_data.accumulated_response_ids[:-a_t_len]
                            agent_data.accumulated_response_mask = agent_data.accumulated_response_mask[:-a_t_len]
                            agent_data.accumulated_logprobs = agent_data.accumulated_logprobs[:-a_t_len]
                        agent_data.pending_turn = None
                        self._save_current_trajectory_without_current_turn(agent_data)
                        return await self._trigger_echo_selection(agent_data)
                    else:
                        # SUPO: 回滚当前 turn 的 a_t（保持原逻辑不变）
                        a_t_len = len(agent_data.response_ids)
                        if a_t_len > 0:
                            agent_data.accumulated_response_ids = agent_data.accumulated_response_ids[:-a_t_len]
                            agent_data.accumulated_response_mask = agent_data.accumulated_response_mask[:-a_t_len]
                            agent_data.accumulated_logprobs = agent_data.accumulated_logprobs[:-a_t_len]
                        self._save_current_trajectory_without_current_turn(agent_data)
                        return await self._trigger_summary(agent_data)
                else:
                    agent_data.overlong = True
                    logger.warning(f"Reached max compression rounds ({self.max_summary_rounds}). Marking overlong.")
                    return AgentState.TERMINATED
                    

        # 正常流程（未超限或未启用SUPO）
        if len(agent_data.response_mask) + len(tool_response_ids) >= self.response_length:
            return AgentState.TERMINATED

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        # SUPO: 累积工具结果到当前 trajectory（注意 mask=0）
        if self.enable_summarization:
            agent_data.accumulated_response_ids.extend(tool_response_ids)
            agent_data.accumulated_response_mask.extend([0] * len(tool_response_ids))  # tool结果mask=0
            agent_data.accumulated_logprobs.extend([0.0] * len(tool_response_ids))

        agent_data.prompt_ids += tool_response_ids
        agent_data.response_mask += [0] * len(tool_response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(tool_response_ids)
        agent_data.user_turns += 1
        # Terminate if any tool is a stop/finish tool
        if any(name.lower() in ("finish", "stop", "submit") for name in tool_call_names):
            return AgentState.TERMINATED
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        response_ids = await self.apply_chat_template(
            add_messages,
            remove_system_prompt=True,
        )

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    def _initialize_interactions(self, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        return interaction_map

    def _extract_turn_finding(self, full_text: str) -> str:
        """ECHO: 从完整 response 文本中提取 <sum_last_turn> 摘要。"""
        # 1. 完整标签（全文搜索，兼容 think 内外）
        m = re.search(r'<sum_last_turn>(.*?)</sum_last_turn>', full_text, flags=re.DOTALL)
        if m and m.group(1).strip():
            return m.group(1).strip()[:300]
        # 2. 有开标签没闭标签（被截断）
        m = re.search(r'<sum_last_turn>(.*)', full_text, flags=re.DOTALL)
        if m and m.group(1).strip():
            return m.group(1).strip()[:300]
        # 没写标签：返回空，由 _build_echo_prompt 用截断 tool_response 兜底
        return ""

    def _parse_selection_indices(self, full_output: str, max_turns: int) -> list[dict] | None:
        """ECHO: 从 actor 输出中解析选中的 turn indices。
        返回 [{"index": int, "score": float}, ...] 或 None（解析失败）。"""
        # 去掉 think 块
        cleaned = re.sub(r'<think>.*?</think>', '', full_output, flags=re.DOTALL).strip()
        # 从 <selection>...</selection> 标签中提取
        selection_match = re.search(r'<selection>(.*?)</selection>', cleaned, flags=re.DOTALL)
        if not selection_match:
            logger.warning(f"ECHO: No <selection> tag found in output: {repr(cleaned[:200])}")
            return None

        content = selection_match.group(1).strip()
        results = []
        seen = set()
        out_of_range = []
        # 匹配 turn_N 格式（不会误匹配年份等数字）
        for m in re.finditer(r'turn_(\d+)', content, re.IGNORECASE):
            idx = int(m.group(1))
            if idx in seen:
                continue
            seen.add(idx)
            if idx < max_turns:
                results.append({"index": idx, "score": 0.5})
            else:
                out_of_range.append(idx)
        if out_of_range:
            logger.warning(
                f"ECHO: Out-of-range turn indices {out_of_range} (max={max_turns-1}) ignored. "
                f"Valid selections={[r['index'] for r in results]}. content={content[:150]!r}"
            )
        # 空 selection 是合法的：模型选择不保留任何历史轮（pending 已自动保留）
        return sorted(results, key=lambda x: x["index"])

    def _build_echo_prompt(self, agent_data: AgentData, selected_turns: list[dict], pending_turn: dict = None) -> list[dict]:
        """ECHO: 用选中的 turns + pending_turn（自动保留）构建新 prompt messages
        selected_turns: [{"index": int, "score": float}, ...] — 从 turn_history 中选出的历史轮
        pending_turn: 最新轮（自动保留，不参与选择），可能有 finding
        """
        selected_messages = []
        for item in selected_turns:
            idx = item["index"]
            if idx < len(agent_data.turn_history):
                turn = agent_data.turn_history[idx]
                query = turn.get("query", "")
                finding = turn.get("finding", "")
                selected_messages.append({"role": "assistant", "content": f'search({{"query": "{query}"}})'})
                if finding:
                    selected_messages.append({"role": "tool", "content": f"[turn_{idx} Key Finding] {finding}"})
                else:
                    tr_text = turn.get("tool_response_text", "")[:200]
                    selected_messages.append({"role": "tool", "content": tr_text})

        # 自动追加 pending_turn（最新轮）
        if pending_turn:
            query = pending_turn.get("query", "")
            finding = pending_turn.get("finding", "")
            selected_messages.append({"role": "assistant", "content": f'search({{"query": "{query}"}})'})
            if finding:
                selected_messages.append({"role": "tool", "content": f"[Key Finding] {finding}"})
            elif pending_turn.get("tool_response_text"):
                tr_text = pending_turn.get("tool_response_text", "")[:200]
                selected_messages.append({"role": "tool", "content": tr_text})
            # 否则：既无 finding 也无 tool_response，不 append tool message（本轮结果因 overflow 被丢弃）

        new_messages = [
            *agent_data.original_messages,
            *selected_messages,
            {"role": "user", "content": "The above are the most relevant turns from your previous interaction. Please continue solving the problem based on this context. You may call tools as needed."},
        ]
        selected_indices = [item["index"] for item in selected_turns]
        pending_info = f", pending={'with finding' if pending_turn and pending_turn.get('finding') else 'no finding'}" if pending_turn else ""
        logger.debug(f"ECHO: Reconstructed prompt with turns {selected_indices} out of {len(agent_data.turn_history)}{pending_info}")
        return new_messages

    def _summary_exclude_think(self, full_output):
        """
        排除full_output中的<think>思考内容</think>，提取summary内容。
        如果提取失败（截断等），对 fallback 内容做长度上限保护。
        """
        # 1. 去掉 think 块（兼容换行）
        cleaned = re.sub(r'<think>.*?</think>', '', full_output, flags=re.DOTALL).strip()

        # 2. 提取 summary 标签内的纯内容
        match = re.search(r'<summary>(.*?)</summary>', cleaned, flags=re.DOTALL)
        if match:
            summary_content = match.group(1).strip()
        else:
            # fallback：尝试提取 <summary> 开标签后的内容（处理截断导致无闭标签的情况）
            open_match = re.search(r'<summary>(.*)', cleaned, flags=re.DOTALL)
            if open_match:
                summary_content = open_match.group(1).strip()
            else:
                summary_content = cleaned
            logger.warning("SUPO: No <summary> tag found in output, using fallback")
            # 防护：fallback 内容截断到合理长度（避免下一轮 prompt 被左截断）
            max_summary_chars = 3072
            if len(summary_content) > max_summary_chars:
                summary_content = summary_content[:max_summary_chars] + "...(truncated)"
                logger.warning(f"SUPO: Fallback summary truncated to {max_summary_chars} chars")

        return summary_content

    async def _trigger_summary(self, agent_data: AgentData) -> AgentState:
        """SUPO: 触发 summary generation"""
        summary_messages = [{"role": "user", "content": self.summary_instruction}]
        summary_instruction_ids = await self.apply_chat_template(
            summary_messages, remove_system_prompt=True,
        )
        agent_data.sum_instruction_ids = summary_instruction_ids
        agent_data.prompt_ids = (
            agent_data.current_traj_prompt_ids +
            agent_data.accumulated_response_ids +
            summary_instruction_ids
        )
        agent_data.response_mask = []
        agent_data.response_logprobs = []
        agent_data.is_summarizing = True
        logger.debug(f"SUPO: Triggering summarization. summary_count={agent_data.summary_count}")
        return AgentState.GENERATING

    async def _trigger_echo_selection(self, agent_data: AgentData) -> AgentState:
        """ECHO: 触发 turn selection generation
        pending_turn（最新轮）单独展示让模型 summary，不参与选择；
        turn_history（已有 finding 的历史轮）参与选择。
        首轮超长时 turn_history 为空，只做 summary 不做 selection。
        """
        # 构建历史 turn 列表（都有 finding）
        turn_descriptions = []
        for i, turn in enumerate(agent_data.turn_history):
            query = turn.get("query", "")[:100]
            finding = turn.get("finding", "")[:150]
            turn_descriptions.append(f"turn_{i}: query=\"{query}\" | finding=\"{finding}\"")
        turn_list_str = "\n".join(turn_descriptions)

        # pending_turn 信息（最新轮）
        pending = getattr(agent_data, 'pending_turn', None)
        pending_query = pending.get("query", "")[:100] if pending else ""
        pending_tr = pending.get("tool_response_text", "")[:1500] if pending else ""
        pending_has_finding = bool(pending and pending.get("finding"))

        # 通用流程：始终展示 turn_history（可能为空），统一处理
        if not turn_list_str:
            turn_list_str = "(none)"

        n_hist = len(agent_data.turn_history)
        idx_range_str = f"turn_0..turn_{n_hist - 1}" if n_hist > 0 else "(none)"
        instruction = self.selection_instruction.format(turn_list=turn_list_str)
        instruction += f"\n\nValid selection indices: {idx_range_str}. The latest turn is auto-kept and is NOT a turn_N."
        if pending_has_finding:
            # 2b 优化：pending 已有 finding，只做 selection
            instruction += f"\nLatest turn (auto-kept): \"{pending_query}\" -> {pending.get('finding', '')[:200]}"
        elif not pending_tr:
            # tool_response 因 overflow 被丢弃，无内容可 summarize，只做 selection
            instruction += "\nLatest turn is auto-kept (tool response discarded due to context limit)."
        else:
            # 标准流程：sum 在前，selection 在后
            instruction += (
                f"\nLatest turn (auto-kept):\n  query=\"{pending_query}\"\n  tool_response:\n{pending_tr}\n\n"
                f"Summarize it inside <sum_last_turn></sum_last_turn>, then output your selection."
            )
        # 统一添加 "exactly once" 要求，避免模型重复生成 selection
        instruction += "\nDo this exactly once. Do NOT repeat the selection."
        # 构建独立 prompt，不带原始对话上下文，避免模型被原始任务格式干扰
        selection_messages = [{"role": "user", "content": instruction}]
        selection_instruction_ids = await self.apply_chat_template(
            selection_messages, remove_system_prompt=False,
        )
        agent_data.sum_instruction_ids = selection_instruction_ids  # 复用字段，用于追加到 trajectory
        # 关键：prompt 只包含选择指令，不拼接原始对话，让模型专注于选 turn
        agent_data.prompt_ids = selection_instruction_ids
        agent_data.response_mask = []
        agent_data.response_logprobs = []
        agent_data.is_summarizing = True  # 复用 is_summarizing 标记，表示正在做上下文压缩
        logger.warning(
            f"[ECHO-DBG trigger] req={agent_data.request_id} sum_cnt={agent_data.summary_count} "
            f"turn_history_len={len(agent_data.turn_history)} pending_has_finding={pending_has_finding} "
            f"traj_count={len(agent_data.trajectory_outputs)}"
        )
        return AgentState.GENERATING

    # SUPO/ECHO support
    def _save_current_trajectory_without_current_turn(self, agent_data: AgentData):
        """
        SUPO: 保存当前trajectory，此时 accumulated_response_ids 已经移除了当前轮的a_t
        """
        resp = agent_data.accumulated_response_ids
        mask = agent_data.accumulated_response_mask
        logp = agent_data.accumulated_logprobs
        
        # 确保三者同长
        min_len = min(len(resp), len(mask), len(logp))
        
        agent_data.trajectory_outputs.append(TrajectoryOutput(
            prompt_ids=list(agent_data.current_traj_prompt_ids),
            response_ids=list(resp[:min_len]),
            response_mask=list(mask[:min_len]),
            response_logprobs=list(logp[:min_len]),
            is_final=False,
        ))
        

    # 修改 4: _append_summary_to_current_trajectory - 确保正确追加
    def _append_summary_to_current_trajectory(self, agent_data: AgentData, summary_logprobs: list[float]):
        """
        SUPO: 将摘要指令和摘要内容追加到最后一个trajectory
        """
        if not agent_data.trajectory_outputs:
            logger.warning("SUPO: No trajectory to append summary to!")
            return

        last_traj = agent_data.trajectory_outputs[-1]
        prev_len = len(last_traj.response_ids)

        # 追加 sum_instruction 到 response（mask=0，无梯度）
        if hasattr(agent_data, 'sum_instruction_ids') and agent_data.sum_instruction_ids:
            last_traj.response_ids.extend(agent_data.sum_instruction_ids)
            last_traj.response_mask.extend([0] * len(agent_data.sum_instruction_ids))
            last_traj.response_logprobs.extend([0.0] * len(agent_data.sum_instruction_ids))

        # 追加摘要内容（mask=1，有梯度）
        last_traj.response_ids.extend(agent_data.response_ids)
        last_traj.response_mask.extend([1] * len(agent_data.response_ids))
        resp_len = len(agent_data.response_ids)
        if summary_logprobs:
            last_traj.response_logprobs.extend(summary_logprobs[:resp_len])
        else:
            last_traj.response_logprobs.extend([0.0] * resp_len)

        decoded_traj = self.tokenizer.decode(last_traj.response_ids, skip_special_tokens=True)
        sel_in_traj = decoded_traj.count("<selection>")
        _prev_ids = last_traj.response_ids[:prev_len]
        _inst_ids = agent_data.sum_instruction_ids if hasattr(agent_data, "sum_instruction_ids") else []
        _model_ids = agent_data.response_ids
        _sel_in_prev = self.tokenizer.decode(_prev_ids, skip_special_tokens=True).count("<selection>")
        _sel_in_inst = self.tokenizer.decode(_inst_ids, skip_special_tokens=True).count("<selection>") if _inst_ids else 0
        _sel_in_model = self.tokenizer.decode(_model_ids, skip_special_tokens=True).count("<selection>")
        logger.warning(
            f"[ECHO-DBG append] req={agent_data.request_id} traj_idx={len(agent_data.trajectory_outputs)-1} "
            f"prev_traj_resp_len={prev_len} "
            f"instruction_len={len(_inst_ids)} "
            f"summary_len={len(agent_data.response_ids)} "
            f"new_traj_resp_len={len(last_traj.response_ids)} "
            f"sel_in_traj={sel_in_traj} "
            f"sel_breakdown=prev:{_sel_in_prev}+inst:{_sel_in_inst}+model:{_sel_in_model}"
        )
        if _sel_in_prev > 0:
            _prev_text = self.tokenizer.decode(_prev_ids, skip_special_tokens=True)
            _pos = _prev_text.find("<selection>")
            logger.warning(
                f"[ECHO-DBG append] req={agent_data.request_id} PREV_HAS_SELECTION at char {_pos}, "
                f"ctx={_prev_text[max(0,_pos-100):_pos+200]!r}"
            )
        if _sel_in_inst > 1:
            _inst_text = self.tokenizer.decode(_inst_ids, skip_special_tokens=True)
            # find each <selection> and dump context
            _starts = []
            _cursor = 0
            while True:
                _p = _inst_text.find("<selection>", _cursor)
                if _p < 0:
                    break
                _starts.append(_p)
                _cursor = _p + 1
            for _idx, _p in enumerate(_starts):
                logger.warning(
                    f"[ECHO-DBG append] req={agent_data.request_id} INST_SEL#{_idx} at char {_p}, "
                    f"ctx={_inst_text[max(0,_p-60):_p+100]!r}"
                )


    def _save_final_trajectory(self, agent_data: AgentData):
        """
        SUPO: 保存最终trajectory，包含当前累积的所有response
        """
        agent_data.trajectory_outputs.append(TrajectoryOutput(
            prompt_ids=list(agent_data.current_traj_prompt_ids),
            response_ids=list(agent_data.accumulated_response_ids),
            response_mask=list(agent_data.accumulated_response_mask),
            response_logprobs=list(agent_data.accumulated_logprobs),
            is_final=True,
        ))
        
        logger.debug(
            f"SUPO: Saved final trajectory {len(agent_data.trajectory_outputs)-1}. "
            f"prompt_len={len(agent_data.current_traj_prompt_ids)}, "
            f"response_len={len(agent_data.accumulated_response_ids)}"
        )
