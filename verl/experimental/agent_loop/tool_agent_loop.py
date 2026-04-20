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
        self.summary_count: int = 0  # 已执行的摘要轮次 I
        self.trajectory_outputs: list[TrajectoryOutput] = []  # 累积的子轨迹输出
        self.is_summarizing: bool = False  # 当前是否在生成摘要（v_sum ∈ s_t）
        self.overlong: bool = False  # 是否因超长被终止
        # 当前trajectory累积的response（用于保存trajectory）
        self.current_traj_prompt_ids: list[int] = []
        self.accumulated_response_ids: list[int] = []
        self.accumulated_response_mask: list[int] = []
        self.accumulated_logprobs: list[float] = []


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
        self.summary_instruction = getattr(
            self.rollout_config.multi_turn, 
            'summary_instruction', 
            "System:\nYou are a helpful agent interacting with a function calling environment to solve user's problem. The interaction history is now too long. Please summarize the interaction history.\n• Remember to keep the important information in the history to ensure that you can continue solving the problem.\n• Do not call any function in this turn.\nNow generate the summary, and put your summary inside tag <summary></summary>.\n"
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
        
        # SUPO: 摘要生成不增加 assistant_turns
        if not (self.enable_summarization and agent_data.is_summarizing):
            agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # 处理摘要生成阶段 
        if self.enable_summarization and agent_data.is_summarizing:
            # 追加摘要到当前 trajectory
            self._append_summary_to_current_trajectory(agent_data, output.log_probs or [])

            # 解码 summary 文本用于构造新 messages
            full_output = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            summary_text = self._summary_exclude_think(full_output)

            new_messages = [
                *agent_data.original_messages,  # 包含原始的 system 和 user
                {"role": "assistant", "content": f"<summary>{summary_text}</summary>"},
                {"role": "user", "content": "The above is a summary of your previous interaction history. Please continue solving the problem based on this context. You may call tools as needed."},
            ]

            new_prompt_ids = await self.apply_chat_template(new_messages, tools=self.tool_schemas)
            agent_data.prompt_ids = new_prompt_ids  # 必须更新全局 prompt_ids
            agent_data.messages = new_messages      # 建议也更新 messages 状态
            
            agent_data.current_traj_prompt_ids = list(agent_data.prompt_ids)
            # 清空历史累积
            agent_data.accumulated_response_ids = []
            agent_data.accumulated_response_mask = []
            agent_data.accumulated_logprobs = []
            
            # 重置标记
            agent_data.is_summarizing = False
            agent_data.summary_count += 1

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

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)
        # Determine next state
        if agent_data.tool_calls:
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

        agent_data.messages.extend(add_messages)

        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            tool_response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            # Note that we have to pass None to the images and videos if there are no new images / videos
            # to stay compatible with downstream image processing logic!
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            tool_response_ids = await self.apply_chat_template(
                add_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

        # ========== MODIFIED: 正确的 SUPO 上下文超限检测 ==========
        if self.enable_summarization:
            # 总长度 = prompt + 累积response + 当前a_t + 工具结果o_t
            current_total_length = (
                len(agent_data.prompt_ids) +           # 基础 prompt (s_1 + 可能的前一个summary)
                len(agent_data.accumulated_response_ids) +  # 历史累积的 response
                len(tool_response_ids)                 # 工具返回 o_t
            )
            
            if current_total_length >= self.working_context_length:
                logger.debug(
                    f"SUPO: Context overflow detected. "
                    f"total={current_total_length} >= L={self.working_context_length}. "
                    f"Discarding current turn (a_t, o_t)."
                )
                
                if agent_data.summary_count < self.max_summary_rounds:
                    # 关键修改：使用 pointer 回滚到 trajectory 起始位置（安全，不受编码长度变化影响）
                    a_t_len = len(agent_data.response_ids)
                    if a_t_len > 0:
                        agent_data.accumulated_response_ids = agent_data.accumulated_response_ids[:-a_t_len]
                        agent_data.accumulated_response_mask = agent_data.accumulated_response_mask[:-a_t_len]
                        agent_data.accumulated_logprobs = agent_data.accumulated_logprobs[:-a_t_len]
                    
                    # 保存当前 trajectory（不含当前轮 a_t 和 o_t）
                    self._save_current_trajectory_without_current_turn(agent_data)
                    
                    # 准备摘要指令
                    summary_messages = [{"role": "user", "content": self.summary_instruction}]
                    summary_instruction_ids = await self.apply_chat_template(
                        summary_messages,
                        remove_system_prompt=True,
                    )
                    agent_data.sum_instruction_ids = summary_instruction_ids
                    
                    # 关键修改：设置生成用的 prompt
                    # 包含：基础prompt + 累积的历史（已移除a_t）+ 摘要指令
                    agent_data.prompt_ids = (
                        agent_data.current_traj_prompt_ids + 
                        agent_data.accumulated_response_ids + 
                        summary_instruction_ids
                    )
                    
                    # 重置状态
                    agent_data.response_mask = []
                    agent_data.response_logprobs = []
                    agent_data.is_summarizing = True
                    
                    logger.debug(f"SUPO: Triggering summarization. summary_count={agent_data.summary_count}")
                    return AgentState.GENERATING
                else:
                    # 达到最大摘要次数
                    agent_data.overlong = True
                    logger.warning(f"SUPO: Reached max summary rounds. Marking as overlong.")
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

    def _summary_exclude_think(self, full_output):
        """
        排除full_output中的<think>思考内容</think>
        """
                # 1. 去掉 think 块（兼容换行）
        cleaned = re.sub(r'<think>.*?</think>', '', full_output, flags=re.DOTALL).strip()

        # 2. 提取 summary 标签内的纯内容
        match = re.search(r'<summary>(.*?)</summary>', cleaned, flags=re.DOTALL)
        if match:
            summary_content = match.group(1).strip()
        else:
            # fallback：如果没找到 summary 标签，用清理后的全文
            summary_content = cleaned
            logger.warning("SUPO: No <summary> tag found in output")

        return summary_content

    # SUPO support
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
        
        # 追加 sum_instruction 到 response（mask=0，无梯度）
        if hasattr(agent_data, 'sum_instruction_ids') and agent_data.sum_instruction_ids:
            last_traj.response_ids.extend(agent_data.sum_instruction_ids)
            last_traj.response_mask.extend([0] * len(agent_data.sum_instruction_ids))
            last_traj.response_logprobs.extend([0.0] * len(agent_data.sum_instruction_ids))

        # 追加摘要内容（mask=1，有梯度）
        last_traj.response_ids.extend(agent_data.response_ids)
        last_traj.response_mask.extend([1] * len(agent_data.response_ids))
        if summary_logprobs:
            last_traj.response_logprobs.extend(summary_logprobs)
        else:
            last_traj.response_logprobs.extend([0.0] * len(agent_data.response_ids))

        logger.debug(
            f"SUPO: Appended sum_instruction + summary to trajectory {len(agent_data.trajectory_outputs)-1}. "
            f"instruction_len={len(agent_data.sum_instruction_ids) if hasattr(agent_data, 'sum_instruction_ids') else 0}, "
            f"summary_len={len(agent_data.response_ids)}"
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
