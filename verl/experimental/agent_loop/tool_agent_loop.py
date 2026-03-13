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
import logging
import os
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
from verl.experimental.agent_loop.memory_graph import MemoryGraph, build_scorer
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

        # Memory graph for causal context management (ECHO)
        self.memory_graph: Optional[MemoryGraph] = None
        # turn_ids in the response portion of the current sub-trajectory (Trace to Learn)
        self.turn_ids: list[int] = []
        # Pending assistant text/tokens for the current turn (set in GENERATING, consumed in PROCESSING_TOOLS)
        self._pending_think_text: str = ""
        self._pending_think_token_len: int = 0
        # Think token ids staged from GENERATING, consumed in PROCESSING_TOOLS / INTERACTING
        self._current_turn_think_tokens: list[int] = []

        # Snapshot of initial prompt token ids (system + user), saved once in PENDING state.
        # Used to rebuild prompt_ids after context pruning.
        self.initial_prompt_ids: list[int] = []
        # Per-turn token ids: turn_token_ids[i] = think_tokens + tool_result_tokens for turn i.
        # Used internally to rebuild prompt_ids; NOT exported to training side.
        self.turn_token_ids: list[list[int]] = []

        # Completed sub-trajectories captured at each context management trigger.
        # Each entry: {
        #   "prompt_ids":       list[int],   # pruned context for this sub-traj
        #   "response_ids":     list[int],   # trainable tokens
        #   "response_mask":    list[int],   # 1=trainable, 0=context
        #   "response_logprobs": list[float],
        #   "response_turn_ids": list[int],  # memory-graph turn_ids (Trace to Learn)
        # }
        # The final (ongoing) sub-trajectory is the main AgentLoopOutput; it is NOT in this list.
        self.completed_sub_trajectories: list[dict] = []


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

        # ECHO memory graph config
        memory_graph_cfg = getattr(self.rollout_config.multi_turn, "memory_graph", None)
        self.memory_graph_enabled = memory_graph_cfg is not None and memory_graph_cfg.enabled
        if self.memory_graph_enabled:
            self.memory_graph_scorer = build_scorer(
                scorer_type=memory_graph_cfg.scorer_type,
                model_name=memory_graph_cfg.model_name,
                device=memory_graph_cfg.device,
                api_base_url=memory_graph_cfg.api_base_url,
                api_key=memory_graph_cfg.api_key,
                api_model=memory_graph_cfg.api_model,
            )
            self.memory_graph_top_k = memory_graph_cfg.top_k
            self.memory_graph_strategy = memory_graph_cfg.strategy
            self.memory_graph_short_memory_turns = memory_graph_cfg.short_memory_turns
            self.single_turn_max_tokens = memory_graph_cfg.single_turn_max_tokens
            # context_max_tokens=None means "use prompt_length as threshold"
            self.context_max_tokens = memory_graph_cfg.context_max_tokens or self.prompt_length

            # Summarizer API config (OpenAI-compatible). If not set, falls back to truncation.
            summarizer_cfg = memory_graph_cfg.summarizer
            if summarizer_cfg is not None:
                try:
                    from openai import AsyncOpenAI
                    self._summarizer_client = AsyncOpenAI(
                        base_url=summarizer_cfg.api_base_url,
                        api_key=summarizer_cfg.api_key,
                    )
                    self._summarizer_model = summarizer_cfg.api_model
                except ImportError:
                    logger.warning("openai package not found; summarizer will fall back to truncation.")
                    self._summarizer_client = None
                    self._summarizer_model = None
            else:
                self._summarizer_client = None
                self._summarizer_model = None

        # Initialize interactions from config file
        self.interaction_config_file = self.rollout_config.multi_turn.interaction_config_path
        if self.interaction_config_file:
            self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(
                self.interaction_config_file
            )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
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
        if self.memory_graph_enabled:
            agent_data.memory_graph = MemoryGraph(
                top_k=self.memory_graph_top_k,
                scorer=self.memory_graph_scorer,
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
        if agent_data.memory_graph is not None:
            output.extra_fields["memory_graph_adjacency"] = agent_data.memory_graph.get_adjacency()
            output.extra_fields["turn_ids"] = agent_data.turn_ids
            output.extra_fields["completed_sub_trajectories"] = agent_data.completed_sub_trajectories
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
        if agent_data.memory_graph is not None:
            agent_data.initial_prompt_ids = list(prompt_ids)
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )
        # first time to set num_preempted
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        # then add num_preempted to the metrics
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            # Multi-round calls, only update the maximum max_global_steps.
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Stage assistant text/tokens for memory graph node (consumed in PROCESSING_TOOLS/INTERACTING)
        if agent_data.memory_graph is not None:
            agent_data._pending_think_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            agent_data._pending_think_token_len = len(agent_data.response_ids)
            agent_data._current_turn_think_tokens = list(agent_data.response_ids)

        # Check termination conditions
        terminated = False
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            terminated = True
        elif self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            terminated = True
        elif self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            terminated = True

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
        if terminated or (not agent_data.tool_calls and not self.interaction_config_file):
            # Terminal turn: create node with think only (no tool_result)
            if agent_data.memory_graph is not None:
                node = agent_data.memory_graph.add_node(
                    think_text=agent_data._pending_think_text,
                    tool_result_text="",
                    think_token_len=agent_data._pending_think_token_len,
                    tool_result_token_len=0,
                )
                agent_data.turn_ids.append(node.turn_id)
                agent_data.turn_token_ids.append(list(agent_data._current_turn_think_tokens))
            return AgentState.TERMINATED
        elif agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        else:
            return AgentState.INTERACTING

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

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

        # ECHO: Single-turn summary — if think + tool_result exceeds threshold,
        # summarize tool_result to keep turns compact for better causal scoring.
        # Only tool_result (mask=0) is summarized, training gradients are unaffected.
        if agent_data.memory_graph is not None:
            tool_result_text = " ".join(
                msg.get("content", "") if isinstance(msg.get("content"), str)
                else " ".join(c.get("text", "") for c in msg.get("content", []) if c.get("type") == "text")
                for msg in add_messages
            )
            tool_result_token_len = len(
                await self.loop.run_in_executor(
                    None, lambda txt=tool_result_text: self.tokenizer.encode(txt, add_special_tokens=False)
                )
            )
            if agent_data._pending_think_token_len + tool_result_token_len > self.single_turn_max_tokens:
                tool_result_text = await self._summarize_text(tool_result_text)
                add_messages = [{"role": "tool", "content": tool_result_text}]
                new_images_this_turn = []
                tool_call_names = tool_call_names[:1]  # keep aligned with collapsed add_messages

        agent_data.messages.extend(add_messages)

        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            # Note that we have to pass None to the images and videos if there are no new images / videos
            # to stay compatible with downstream image processing logic!
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            response_ids = await self.apply_chat_template(
                add_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1

        # Create memory graph node for this complete turn (assistant think + tool result)
        if agent_data.memory_graph is not None:
            node = agent_data.memory_graph.add_node(
                think_text=agent_data._pending_think_text,
                tool_result_text=tool_result_text,
                think_token_len=agent_data._pending_think_token_len,
                tool_result_token_len=len(response_ids),
            )
            agent_data.turn_ids.append(node.turn_id)
            # Record per-turn token ids for prompt_ids reconstruction
            agent_data.turn_token_ids.append(
                agent_data._current_turn_think_tokens + list(response_ids)
            )

            # ECHO: Context management — if total context exceeds threshold,
            # snapshot current sub-trajectory, then rebuild (compress) prompt_ids.
            if len(agent_data.prompt_ids) > self.context_max_tokens:
                retained = agent_data.memory_graph.get_context_turn_ids(
                    context_max_tokens=self.context_max_tokens,
                    short_memory_turns=self.memory_graph_short_memory_turns,
                    strategy=self.memory_graph_strategy,
                )

                # Snapshot the completed sub-trajectory before compressing
                split_idx = len(agent_data.prompt_ids) - len(agent_data.response_mask)
                agent_data.completed_sub_trajectories.append({
                    "prompt_ids": agent_data.prompt_ids[:split_idx],
                    "response_ids": agent_data.prompt_ids[split_idx:],
                    "response_mask": list(agent_data.response_mask),
                    "response_logprobs": list(agent_data.response_logprobs),
                    "response_turn_ids": list(agent_data.turn_ids),
                })

                # Rebuild prompt_ids with only the retained turns (Prune to Act)
                new_prompt_ids = list(agent_data.initial_prompt_ids)
                for tid in retained:
                    new_prompt_ids += agent_data.turn_token_ids[tid]
                agent_data.prompt_ids = new_prompt_ids

                # Reset response tracking for the new sub-trajectory
                agent_data.response_mask = []
                agent_data.response_logprobs = []
                agent_data.turn_ids = []

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

        # Create memory graph node for this turn (assistant think + interaction response)
        if agent_data.memory_graph is not None:
            node = agent_data.memory_graph.add_node(
                think_text=agent_data._pending_think_text,
                tool_result_text=interaction_responses if isinstance(interaction_responses, str) else str(interaction_responses),
                think_token_len=agent_data._pending_think_token_len,
                tool_result_token_len=len(response_ids),
            )
            agent_data.turn_ids.append(node.turn_id)
            agent_data.turn_token_ids.append(
                agent_data._current_turn_think_tokens + list(response_ids)
            )

        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _summarize_text(self, text: str) -> str:
        """Summarize long tool result text to keep context compact.

        Uses an OpenAI-compatible API when summarizer config is provided;
        falls back to truncation otherwise.
        """
        max_tokens = self.single_turn_max_tokens // 2  # leave room for think portion

        if self._summarizer_client is not None:
            prompt = (
                "Summarize the following tool execution result concisely. "
                "Keep all key facts, numbers, and conclusions. "
                f"Reply in at most {max_tokens} tokens.\n\n"
                f"Tool Result:\n{text}\n\nSummary:"
            )
            try:
                response = await self._summarizer_client.chat.completions.create(
                    model=self._summarizer_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning("Summarizer API call failed (%s); falling back to truncation.", e)

        # Truncation fallback
        tokens = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.encode(text, add_special_tokens=False)
        )
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        summarized = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(tokens, skip_special_tokens=True)
        )
        return summarized + "...(truncated)"

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
