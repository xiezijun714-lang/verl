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
    """A trajectory segment emitted by SUPO-style context compression."""
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float]
    response_turn_ids: list[int]
    response_finding_turn_ids: list[int]
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

        # State for SUPO-style context compression.
        self.original_messages: list[dict] = []
        self.original_prompt_ids: list[int] = []
        self.summary_count: int = 0
        self.trajectory_outputs: list[TrajectoryOutput] = []
        self.is_summarizing: bool = False
        self.overlong: bool = False
        # Response tokens accumulated for the current trajectory segment.
        self.current_traj_prompt_ids: list[int] = []
        self.accumulated_response_ids: list[int] = []
        self.accumulated_response_mask: list[int] = []
        self.accumulated_logprobs: list[float] = []
        self.accumulated_response_turn_ids: list[int] = []
        self.accumulated_response_finding_turn_ids: list[int] = []

        # Turn-level history used by ECHO context reconstruction.
        self.turn_history: list[dict] = []  # [{"query": str, "finding": str, "source_traj_idx": int, "turn_id": int}]
        self.current_selected_traj_indices: list[int] = []
        self.current_selected_turn_ids: list[int] = []
        self.next_turn_id: int = 0


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

        # SUPO/ECHO context compression configuration.
        self.enable_summarization = getattr(self.rollout_config.multi_turn, 'enable_summarization', False)
        self.max_summary_rounds = getattr(self.rollout_config.multi_turn, 'max_summary_rounds', 2)
        self.working_context_length = getattr(self.rollout_config.multi_turn, 'working_context_length', 8192)
        # "summary" uses generated summaries; "echo_e2e" lets the actor select retained turns.
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

        # SUPO can emit multiple trajectory segments for one rollout.
        if self.enable_summarization:
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
            final_selected_traj_indices = (
                sorted(set(agent_data.current_selected_traj_indices))
                if self.context_compression_method == "echo_e2e"
                else []
            )
            final_selected_turn_ids = (
                sorted(set(agent_data.current_selected_turn_ids))
                if self.context_compression_method == "echo_e2e"
                else []
            )
            
            for i, traj in enumerate(agent_data.trajectory_outputs):
                # Truncate prompt and response to fit within max_model_len
                # prompt_ids: take last prompt_length tokens if too long (left truncate)
                # response_ids: take first response_length tokens if too long (right truncate)
                truncated_prompt_ids = traj.prompt_ids[-self.prompt_length:] if len(traj.prompt_ids) > self.prompt_length else traj.prompt_ids
                truncated_response_ids = traj.response_ids[:self.response_length]
                truncated_response_mask = traj.response_mask[:self.response_length]
                truncated_logprobs = traj.response_logprobs[:self.response_length] if traj.response_logprobs else None
                truncated_response_turn_ids = traj.response_turn_ids[:self.response_length]
                truncated_response_finding_turn_ids = traj.response_finding_turn_ids[:self.response_length]
                
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
                        "echo_selected_traj_indices": final_selected_traj_indices if traj.is_final else None,
                        "echo_selected_turn_ids": final_selected_turn_ids if traj.is_final else None,
                        "echo_response_turn_ids": truncated_response_turn_ids,
                        "echo_response_finding_turn_ids": truncated_response_finding_turn_ids,
                    },
                )
                outputs.append(output)
            
            logger.debug(
                "Context-compressed rollout completed. "
                f"num_trajectories={len(outputs)}, overlong={agent_data.overlong}, "
                f"summary_count={agent_data.summary_count}"
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
        # Add the ECHO turn-summary requirement to the system prompt.
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
        
        # Keep the original prompt so compressed contexts can be rebuilt from it.
        if self.enable_summarization:
            agent_data.original_prompt_ids = list(prompt_ids)
            agent_data.original_messages = list(agent_data.messages)
            # The current segment keeps history in the response side so generated tokens remain trainable.
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

        # Compression generations are auxiliary and do not count as assistant turns.
        if not (self.enable_summarization and agent_data.is_summarizing):
            agent_data.assistant_turns += 1

        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Handle the auxiliary compression generation.
        if self.enable_summarization and agent_data.is_summarizing:
            self._append_summary_to_current_trajectory(agent_data, output.log_probs or [])

            full_output = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )

            compression_ok = True
            selected_indices: list = []
            pending = None
            if self.context_compression_method == "echo_e2e":
                pending = getattr(agent_data, 'pending_turn', None)
                if pending:
                    finding = self._extract_turn_finding(full_output)
                    if finding:
                        pending["finding"] = finding
                    self._mark_last_trajectory_finding_tokens(agent_data, full_output, pending.get("turn_id"))

                if agent_data.turn_history:
                    parsed = self._parse_selection_indices(full_output, len(agent_data.turn_history))
                    if parsed is None:
                        logger.debug(
                            "ECHO selection parse failed; retaining all prior turns. "
                            f"num_turns={len(agent_data.turn_history)}"
                        )
                        parsed = [{"index": i, "score": 0.5} for i in range(len(agent_data.turn_history))]
                    selected_indices = parsed
                    new_messages = self._build_echo_prompt(agent_data, selected_indices)
                else:
                    new_messages = self._build_echo_prompt(agent_data, [])
            else:
                summary_text = self._summary_exclude_think(full_output)
                if len(summary_text.strip()) < 10:
                    logger.warning(f"Context summary is too short; text={summary_text[:50]!r}")
                    compression_ok = False
                else:
                    new_messages = [
                        *agent_data.original_messages,
                        {"role": "assistant", "content": f"<summary>{summary_text}</summary>"},
                        {"role": "user", "content": "The above is a summary of your previous interaction history. Please continue solving the problem based on this context. You may call tools as needed."},
                    ]

            if not compression_ok:
                agent_data.is_summarizing = False
                agent_data.overlong = True
                logger.warning("Context compression output unusable. Terminating.")
                return AgentState.TERMINATED

            new_prompt_ids = await self.apply_chat_template(new_messages, tools=self.tool_schemas)
            if self.context_compression_method == "echo_e2e":
                agent_data.current_selected_traj_indices = sorted({
                    agent_data.turn_history[item["index"]].get("source_traj_idx")
                    for item in selected_indices
                    if item["index"] < len(agent_data.turn_history)
                    and agent_data.turn_history[item["index"]].get("source_traj_idx") is not None
                })
                agent_data.current_selected_turn_ids = sorted({
                    agent_data.turn_history[item["index"]].get("turn_id")
                    for item in selected_indices
                    if item["index"] < len(agent_data.turn_history)
                    and agent_data.turn_history[item["index"]].get("turn_id") is not None
                })
                if len(new_prompt_ids) > int(self.working_context_length * 0.8):
                    agent_data.is_summarizing = False
                    agent_data.overlong = True
                    logger.warning(
                        "ECHO rebuilt prompt is still too long. "
                        f"prompt_length={len(new_prompt_ids)}, working_context_length={self.working_context_length}"
                    )
                    return AgentState.TERMINATED

            agent_data.prompt_ids = new_prompt_ids
            agent_data.messages = new_messages
            
            agent_data.current_traj_prompt_ids = list(agent_data.prompt_ids)
            agent_data.response_ids = []
            agent_data.response_logprobs = []
            agent_data.accumulated_response_ids = []
            agent_data.accumulated_response_mask = []
            agent_data.accumulated_logprobs = []
            agent_data.accumulated_response_turn_ids = []
            agent_data.accumulated_response_finding_turn_ids = []
            
            agent_data.is_summarizing = False
            agent_data.summary_count += 1
            # Keep turn history aligned with the rebuilt ECHO prompt.
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
            current_turn_id = agent_data.next_turn_id if self.context_compression_method == "echo_e2e" else -1
            response_turn_ids = [current_turn_id] * len(agent_data.response_ids)
            response_finding_turn_ids = [-1] * len(agent_data.response_ids)
            if self.context_compression_method == "echo_e2e":
                full_text = await self.loop.run_in_executor(
                    None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
                )
                summary_span_mask = self._build_summary_span_mask(full_text, agent_data.response_ids)
                response_turn_ids = self._mask_summary_span_turn_ids(response_turn_ids, summary_span_mask)
                pending = getattr(agent_data, "pending_turn", None)
                if pending:
                    response_finding_turn_ids = self._build_finding_turn_ids(
                        full_text, agent_data.response_ids, pending.get("turn_id")
                    )
            agent_data.accumulated_response_turn_ids.extend(response_turn_ids)
            agent_data.accumulated_response_finding_turn_ids.extend(response_finding_turn_ids)

        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        is_last_turn = self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns
        if is_last_turn:
            agent_data.tool_calls = []
            return AgentState.TERMINATED

        # Extract tool calls
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        # Compress when a generated thinking turn is truncated before a valid tool call.
        if output.stop_reason == "length" and not agent_data.tool_calls:
            if self.enable_summarization and not agent_data.is_summarizing \
                    and agent_data.summary_count < self.max_summary_rounds \
                    and len(agent_data.accumulated_response_ids) > 0:
                logger.warning(
                    "Generated turn was truncated before a tool call; rolling back and compressing. "
                    f"response_length={len(agent_data.response_ids)}"
                )
                if self.context_compression_method == "echo_e2e":
                    truncated_text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
                    finding = self._extract_turn_finding(truncated_text)
                    pending = getattr(agent_data, 'pending_turn', None)
                    if finding and pending:
                        pending["finding"] = finding
                        agent_data.turn_history.append(pending)
                        agent_data.pending_turn = None
                        logger.debug("Extracted ECHO finding from a truncated generation.")

                a_t_len = len(agent_data.response_ids)
                if a_t_len > 0:
                    agent_data.accumulated_response_ids = agent_data.accumulated_response_ids[:-a_t_len]
                    agent_data.accumulated_response_mask = agent_data.accumulated_response_mask[:-a_t_len]
                    agent_data.accumulated_logprobs = agent_data.accumulated_logprobs[:-a_t_len]
                    agent_data.accumulated_response_turn_ids = agent_data.accumulated_response_turn_ids[:-a_t_len]
                    agent_data.accumulated_response_finding_turn_ids = (
                        agent_data.accumulated_response_finding_turn_ids[:-a_t_len]
                    )
                if self.context_compression_method == "echo_e2e":
                    self._save_current_trajectory_without_current_turn(agent_data)
                    if not self._has_echo_selection_history(agent_data):
                        agent_data.overlong = True
                        logger.warning(
                            "ECHO generation truncated before reusable turn history was available. Marking overlong."
                        )
                        return AgentState.TERMINATED
                    return await self._trigger_echo_selection(agent_data)
                else:
                    self._save_current_trajectory_without_current_turn(agent_data)
                    return await self._trigger_summary(agent_data)
            else:
                agent_data.overlong = True
                return AgentState.TERMINATED

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

        # Finalize the previous ECHO turn once the next assistant response is available.
        if self.enable_summarization and self.context_compression_method == "echo_e2e":
            pending = getattr(agent_data, 'pending_turn', None)
            if pending:
                full_text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
                if not pending.get("finding"):
                    finding = self._extract_turn_finding(full_text)
                    if finding:
                        pending["finding"] = finding
                self._mark_finding_turn_tokens(agent_data, full_text, pending.get("turn_id"))

                agent_data.turn_history.append(pending)
                agent_data.pending_turn = None

        # Add a separate ECHO hint message so chat-template boundaries remain intact.
        echo_inject_hint = (
            self.enable_summarization
            and self.context_compression_method == "echo_e2e"
        )
        if echo_inject_hint:
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
                "finding": "",
                "source_traj_idx": len(agent_data.trajectory_outputs),
                "turn_id": agent_data.next_turn_id,
            }
            agent_data.next_turn_id += 1
            
            hint_msg = {
                "role": "user",
                "content": "You must summarize the key finding from the last tool response in <sum_last_turn></sum_last_turn> before <tool_call>",
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

        if self.enable_summarization:
            current_total_length = (
                len(agent_data.current_traj_prompt_ids) +
                len(agent_data.accumulated_response_ids) +
                len(tool_response_ids)
            )
            
            if current_total_length >= self.working_context_length:
                if agent_data.summary_count < self.max_summary_rounds:
                    if self.context_compression_method == "echo_e2e":
                        n_hist = len(getattr(agent_data, 'turn_history', []))
                        pending_finding = bool(
                            hasattr(agent_data, 'pending_turn') and agent_data.pending_turn
                            and agent_data.pending_turn.get("finding")
                        )
                        if agent_data.summary_count > 0 and n_hist <= 1 and not pending_finding:
                            agent_data.overlong = True
                            logger.warning(
                                "ECHO context overflow repeated with too little reusable history. "
                                f"turn_history_length={n_hist}, summary_count={agent_data.summary_count}"
                            )
                            self._save_current_trajectory_without_current_turn(agent_data)
                            return AgentState.TERMINATED

                        # Drop the pending turn because its tool response did not fit in the segment.
                        a_t_len = len(agent_data.response_ids)
                        if a_t_len > 0:
                            agent_data.accumulated_response_ids = agent_data.accumulated_response_ids[:-a_t_len]
                            agent_data.accumulated_response_mask = agent_data.accumulated_response_mask[:-a_t_len]
                            agent_data.accumulated_logprobs = agent_data.accumulated_logprobs[:-a_t_len]
                            agent_data.accumulated_response_turn_ids = agent_data.accumulated_response_turn_ids[:-a_t_len]
                            agent_data.accumulated_response_finding_turn_ids = (
                                agent_data.accumulated_response_finding_turn_ids[:-a_t_len]
                            )
                        agent_data.pending_turn = None
                        self._save_current_trajectory_without_current_turn(agent_data)
                        if not self._has_echo_selection_history(agent_data):
                            agent_data.overlong = True
                            logger.warning(
                                "ECHO context overflow occurred before reusable turn history was available. "
                                "Marking overlong."
                            )
                            return AgentState.TERMINATED
                        return await self._trigger_echo_selection(agent_data)
                    else:
                        a_t_len = len(agent_data.response_ids)
                        if a_t_len > 0:
                            agent_data.accumulated_response_ids = agent_data.accumulated_response_ids[:-a_t_len]
                            agent_data.accumulated_response_mask = agent_data.accumulated_response_mask[:-a_t_len]
                            agent_data.accumulated_logprobs = agent_data.accumulated_logprobs[:-a_t_len]
                            agent_data.accumulated_response_turn_ids = agent_data.accumulated_response_turn_ids[:-a_t_len]
                            agent_data.accumulated_response_finding_turn_ids = (
                                agent_data.accumulated_response_finding_turn_ids[:-a_t_len]
                            )
                        self._save_current_trajectory_without_current_turn(agent_data)
                        return await self._trigger_summary(agent_data)
                else:
                    agent_data.overlong = True
                    logger.warning(
                        f"Reached max context compression rounds ({self.max_summary_rounds}). Marking overlong."
                    )
                    return AgentState.TERMINATED
                    

        if len(agent_data.response_mask) + len(tool_response_ids) >= self.response_length:
            return AgentState.TERMINATED

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        # Tool responses are part of the trajectory but do not receive policy gradients.
        if self.enable_summarization:
            agent_data.accumulated_response_ids.extend(tool_response_ids)
            agent_data.accumulated_response_mask.extend([0] * len(tool_response_ids))
            agent_data.accumulated_logprobs.extend([0.0] * len(tool_response_ids))
            tool_turn_id = agent_data.pending_turn["turn_id"] if echo_inject_hint and hasattr(agent_data, "pending_turn") else -1
            agent_data.accumulated_response_turn_ids.extend([tool_turn_id] * len(tool_response_ids))
            agent_data.accumulated_response_finding_turn_ids.extend([-1] * len(tool_response_ids))

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
        """Extract the latest ECHO turn summary from generated text."""
        m = re.search(r'<sum_last_turn>(.*?)</sum_last_turn>', full_text, flags=re.DOTALL)
        if m and m.group(1).strip():
            return m.group(1).strip()[:300]
        m = re.search(r'<sum_last_turn>(.*)', full_text, flags=re.DOTALL)
        if m and m.group(1).strip():
            return m.group(1).strip()[:300]
        return ""

    def _has_echo_selection_history(self, agent_data: AgentData) -> bool:
        """Whether ECHO has at least one retained turn that can participate in selection."""
        return bool(agent_data.turn_history)

    def _parse_selection_indices(self, full_output: str, max_turns: int) -> list[dict] | None:
        """Parse selected ECHO turn indices from actor output."""
        cleaned = re.sub(r'<think>.*?</think>', '', full_output, flags=re.DOTALL).strip()
        selection_match = re.search(r'<selection>(.*?)</selection>', cleaned, flags=re.DOTALL)
        if not selection_match:
            logger.debug("No ECHO <selection> tag found in compression output.")
            return None

        content = selection_match.group(1).strip()
        results = []
        seen = set()
        out_of_range = []
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
                f"ECHO selection contained out-of-range turns {out_of_range}; "
                f"max_turn_index={max_turns - 1}, valid_selections={[r['index'] for r in results]}"
            )
        return sorted(results, key=lambda x: x["index"])

    def _build_echo_prompt(self, agent_data: AgentData, selected_turns: list[dict]) -> list[dict]:
        """Build prompt messages from selected ECHO turn history."""
        selected_messages = []
        for item in selected_turns:
            idx = item["index"]
            if idx < len(agent_data.turn_history):
                turn = agent_data.turn_history[idx]
                query = turn.get("query", "")
                finding = turn.get("finding", "")
                selected_messages.append({"role": "assistant", "content": f"[turn_{idx} Query] {query}"})
                selected_messages.append({"role": "tool", "content": f"[turn_{idx} Key Finding] {finding}"})

        new_messages = [
            *agent_data.original_messages,
            *selected_messages,
            {"role": "user", "content": "The above are the most relevant turns from your previous interaction. Please continue solving the problem based on this context. You may call tools as needed."},
        ]
        selected_indices = [item["index"] for item in selected_turns]
        return new_messages

    def _summary_exclude_think(self, full_output):
        """Extract summary text while ignoring the generated thinking block."""
        cleaned = re.sub(r'<think>.*?</think>', '', full_output, flags=re.DOTALL).strip()

        match = re.search(r'<summary>(.*?)</summary>', cleaned, flags=re.DOTALL)
        if match:
            summary_content = match.group(1).strip()
        else:
            open_match = re.search(r'<summary>(.*)', cleaned, flags=re.DOTALL)
            if open_match:
                summary_content = open_match.group(1).strip()
            else:
                summary_content = cleaned
            logger.warning("No <summary> tag found in compression output; using fallback text.")
            max_summary_chars = 3072
            if len(summary_content) > max_summary_chars:
                summary_content = summary_content[:max_summary_chars] + "...(truncated)"
                logger.warning(f"Fallback summary truncated to {max_summary_chars} characters.")

        return summary_content

    async def _trigger_summary(self, agent_data: AgentData) -> AgentState:
        """Trigger summary-based context compression."""
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
        logger.debug(f"Triggering summary compression. summary_count={agent_data.summary_count}")
        return AgentState.GENERATING

    async def _trigger_echo_selection(self, agent_data: AgentData) -> AgentState:
        """Trigger ECHO turn-selection context compression."""
        turn_descriptions = []
        for i, turn in enumerate(agent_data.turn_history):
            query = turn.get("query", "")
            finding = turn.get("finding", "")
            turn_descriptions.append(f"turn_{i}: query=\"{query}\" | finding=\"{finding}\"")
        turn_list_str = "\n".join(turn_descriptions) if turn_descriptions else "(none)"
        n_hist = len(agent_data.turn_history)

        instruction = self.selection_instruction.format(turn_list=turn_list_str)
        if n_hist > 0:
            instruction += f"\n\nValid selection indices: turn_0 ~ turn_{n_hist-1}."
        else:
            instruction += f"\n\nNo historical turns for selection, return empty inside <selection></selection>."
        
        # Keep the selection instruction separate from the task prompt.
        selection_messages = [{"role": "user", "content": instruction}]
        selection_instruction_ids = await self.apply_chat_template(
            selection_messages, remove_system_prompt=True,
        )

        agent_data.prompt_ids = (
            agent_data.current_traj_prompt_ids +
            agent_data.accumulated_response_ids +
            selection_instruction_ids
        )
        agent_data.sum_instruction_ids = selection_instruction_ids
        agent_data.response_mask = []
        agent_data.response_logprobs = []
        agent_data.is_summarizing = True
        
        return AgentState.GENERATING

    # SUPO/ECHO support
    def _build_summary_span_mask(self, text: str, response_ids: list[int]) -> list[bool]:
        """Identify generated <sum_last_turn> tokens in the current response."""
        summary_span_mask = [False] * len(response_ids)
        if not response_ids:
            return summary_span_mask
        match = re.search(r"<sum_last_turn>.*?(?:</sum_last_turn>|$)", text, flags=re.DOTALL)
        if not match:
            return summary_span_mask

        prefix_ids = self.tokenizer.encode(text[: match.start()], add_special_tokens=False)
        span_ids = self.tokenizer.encode(text[match.start() : match.end()], add_special_tokens=False)
        start = min(len(prefix_ids), len(response_ids))
        end = min(start + len(span_ids), len(response_ids))
        for idx in range(start, end):
            summary_span_mask[idx] = True
        return summary_span_mask

    def _build_finding_turn_ids(self, text: str, response_ids: list[int], turn_id: int | None) -> list[int]:
        """Map the generated <sum_last_turn> span to the contributing ECHO turn."""
        finding_turn_ids = [-1] * len(response_ids)
        if turn_id is None:
            return finding_turn_ids

        summary_span_mask = self._build_summary_span_mask(text, response_ids)
        for idx, is_summary_token in enumerate(summary_span_mask):
            if is_summary_token:
                finding_turn_ids[idx] = int(turn_id)
        return finding_turn_ids

    @staticmethod
    def _mask_summary_span_turn_ids(response_turn_ids: list[int], summary_span_mask: list[bool]) -> list[int]:
        """Remove current-turn credit from summary tokens that belong to a previous turn."""
        return [
            -1 if idx < len(summary_span_mask) and summary_span_mask[idx] else turn_id
            for idx, turn_id in enumerate(response_turn_ids)
        ]

    def _mark_finding_turn_tokens(self, agent_data: AgentData, text: str, turn_id: int | None):
        """Mark the current assistant response's summary tokens for ECHO credit assignment."""
        finding_turn_ids = self._build_finding_turn_ids(text, agent_data.response_ids, turn_id)
        if not any(turn_idx >= 0 for turn_idx in finding_turn_ids):
            return

        start = len(agent_data.accumulated_response_finding_turn_ids) - len(agent_data.response_ids)
        if start < 0:
            return

        for offset, turn_idx in enumerate(finding_turn_ids):
            target_idx = start + offset
            if 0 <= target_idx < len(agent_data.accumulated_response_finding_turn_ids):
                agent_data.accumulated_response_finding_turn_ids[target_idx] = turn_idx
                if target_idx < len(agent_data.accumulated_response_turn_ids):
                    agent_data.accumulated_response_turn_ids[target_idx] = -1

    def _mark_last_trajectory_finding_tokens(self, agent_data: AgentData, text: str, turn_id: int | None):
        """Mark summary tokens appended by a compression generation."""
        if not agent_data.trajectory_outputs:
            return

        finding_turn_ids = self._build_finding_turn_ids(text, agent_data.response_ids, turn_id)
        if not any(turn_idx >= 0 for turn_idx in finding_turn_ids):
            return

        last_traj = agent_data.trajectory_outputs[-1]
        start = len(last_traj.response_finding_turn_ids) - len(agent_data.response_ids)
        if start < 0:
            return

        for offset, turn_idx in enumerate(finding_turn_ids):
            target_idx = start + offset
            if 0 <= target_idx < len(last_traj.response_finding_turn_ids):
                last_traj.response_finding_turn_ids[target_idx] = turn_idx

    def _save_current_trajectory_without_current_turn(self, agent_data: AgentData):
        """Save the current trajectory segment after the truncated turn is removed."""
        resp = agent_data.accumulated_response_ids
        mask = agent_data.accumulated_response_mask
        logp = agent_data.accumulated_logprobs
        turn_ids = agent_data.accumulated_response_turn_ids
        finding_turn_ids = agent_data.accumulated_response_finding_turn_ids
        
        min_len = min(len(resp), len(mask), len(logp), len(turn_ids), len(finding_turn_ids))
        if min_len == 0:
            return
        
        agent_data.trajectory_outputs.append(TrajectoryOutput(
            prompt_ids=list(agent_data.current_traj_prompt_ids),
            response_ids=list(resp[:min_len]),
            response_mask=list(mask[:min_len]),
            response_logprobs=list(logp[:min_len]),
            response_turn_ids=list(turn_ids[:min_len]),
            response_finding_turn_ids=list(finding_turn_ids[:min_len]),
            is_final=False,
        ))
        

    def _append_summary_to_current_trajectory(self, agent_data: AgentData, summary_logprobs: list[float]):
        """Append the compression instruction and generated output to the latest segment."""
        if not agent_data.trajectory_outputs:
            logger.warning("No trajectory segment available for context compression output.")
            return

        last_traj = agent_data.trajectory_outputs[-1]

        if hasattr(agent_data, 'sum_instruction_ids') and agent_data.sum_instruction_ids:
            last_traj.response_ids.extend(agent_data.sum_instruction_ids)
            last_traj.response_mask.extend([0] * len(agent_data.sum_instruction_ids))
            last_traj.response_logprobs.extend([0.0] * len(agent_data.sum_instruction_ids))
            last_traj.response_turn_ids.extend([-1] * len(agent_data.sum_instruction_ids))
            last_traj.response_finding_turn_ids.extend([-1] * len(agent_data.sum_instruction_ids))

        last_traj.response_ids.extend(agent_data.response_ids)
        last_traj.response_mask.extend([1] * len(agent_data.response_ids))
        last_traj.response_turn_ids.extend([-1] * len(agent_data.response_ids))
        last_traj.response_finding_turn_ids.extend([-1] * len(agent_data.response_ids))
        resp_len = len(agent_data.response_ids)
        if summary_logprobs:
            last_traj.response_logprobs.extend(summary_logprobs[:resp_len])
        else:
            last_traj.response_logprobs.extend([0.0] * resp_len)

    def _save_final_trajectory(self, agent_data: AgentData):
        """Save the final trajectory segment."""
        min_len = min(
            len(agent_data.accumulated_response_ids),
            len(agent_data.accumulated_response_mask),
            len(agent_data.accumulated_logprobs),
            len(agent_data.accumulated_response_turn_ids),
            len(agent_data.accumulated_response_finding_turn_ids),
        )
        agent_data.trajectory_outputs.append(TrajectoryOutput(
            prompt_ids=list(agent_data.current_traj_prompt_ids),
            response_ids=list(agent_data.accumulated_response_ids[:min_len]),
            response_mask=list(agent_data.accumulated_response_mask[:min_len]),
            response_logprobs=list(agent_data.accumulated_logprobs[:min_len]),
            response_turn_ids=list(agent_data.accumulated_response_turn_ids[:min_len]),
            response_finding_turn_ids=list(agent_data.accumulated_response_finding_turn_ids[:min_len]),
            is_final=True,
        ))
        
        logger.debug(
            f"Saved final trajectory segment {len(agent_data.trajectory_outputs)-1}. "
            f"prompt_len={len(agent_data.current_traj_prompt_ids)}, "
            f"response_len={len(agent_data.accumulated_response_ids)}"
        )
