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
"""
MemoryGraph: Per-trajectory causal memory graph for ECHO (Prune to Act, Trace to Learn).

Each node represents one complete assistant turn (think + tool_result).
Directed edges represent causal/logical dependency scored by either:
  - CrossEncoderScorer: local cross-encoder model (recommended for quality)
  - APIScorer:          OpenAI-compatible API call (flexible, no local model needed)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

CAUSAL_JUDGE_PROMPT = (
    "Judge the causal dependency between two reasoning steps in an agent trajectory.\n"
    "Step A (earlier):\n{candidate}\n\n"
    "Step B (later):\n{query}\n\n"
    "Does Step B logically depend on the observation or conclusion from Step A?\n"
    "Reply with a single float between 0.0 and 1.0, where 1.0 means strong causal dependency.\n"
    "Output ONLY the number, nothing else."
)


# ---------------------------------------------------------------------------
# Scorer protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class CausalScorer(Protocol):
    """Score causal dependency between a query turn and candidate turns."""

    def score(self, query: str, candidates: list[str]) -> list[float]:
        """
        Return a causal dependency score for each (query, candidate) pair.
        Higher score = stronger causal dependency.
        len(return) == len(candidates).
        """
        ...


# ---------------------------------------------------------------------------
# Scorer implementations
# ---------------------------------------------------------------------------

class CrossEncoderScorer:
    """
    Local cross-encoder model that jointly encodes (query, candidate) pairs
    to judge causal dependency. Best quality for logical relationship detection.

    Recommended: 'cross-encoder/ms-marco-MiniLM-L-6-v2' (~22 MB, CPU ~5-10 ms/pair)

    Install: pip install sentence-transformers
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderScorer. "
                "Install with: pip install sentence-transformers"
            )
        self._model = CrossEncoder(model_name, device=device)

    def score(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        pairs = [[query, c] for c in candidates]
        return self._model.predict(pairs).tolist()


class APIScorer:
    """
    Call an OpenAI-compatible API to judge causal dependency between turns.
    No local model needed — works with any hosted LLM endpoint.

    Args:
        base_url:   API base URL (e.g. "https://api.openai.com/v1").
        api_key:    API key.
        model:      Model name to use (e.g. "gpt-4o-mini").
        prompt_template: Prompt with {query} and {candidate} placeholders.
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "gpt-4o-mini",
        prompt_template: str = CAUSAL_JUDGE_PROMPT,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required for APIScorer. "
                "Install with: pip install openai"
            )
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._prompt_template = prompt_template

    def score(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        scores = []
        for c in candidates:
            prompt = self._prompt_template.format(query=query, candidate=c)
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=8,
                )
                text = resp.choices[0].message.content.strip()
                scores.append(float(text))
            except (ValueError, IndexError, AttributeError) as e:
                logger.warning("APIScorer failed to parse response: %s", e)
                scores.append(0.0)
        return scores


def build_scorer(
    scorer_type: str = "cross_encoder",
    model_name: Optional[str] = None,
    device: str = "cpu",
    api_base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    api_model: Optional[str] = None,
) -> CausalScorer:
    """
    Factory for causal scorers.

    Args:
        scorer_type:  "cross_encoder" | "api"
        model_name:   Override default cross-encoder model name.
        device:       "cpu" or "cuda" (cross_encoder only).
        api_base_url: API base URL (api only).
        api_key:      API key (api only).
        api_model:    API model name (api only).
    """
    if scorer_type == "cross_encoder":
        return CrossEncoderScorer(
            model_name=model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=device,
        )
    elif scorer_type == "api":
        return APIScorer(
            base_url=api_base_url or "https://api.openai.com/v1",
            api_key=api_key or "",
            model=api_model or "gpt-4o-mini",
        )
    else:
        raise ValueError(
            f"Unknown scorer_type: '{scorer_type}'. "
            "Choose from: 'cross_encoder', 'api'."
        )


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class MemoryNode:
    """One node = one complete assistant turn (think + tool_result)."""

    turn_id: int
    think_text: str
    tool_result_text: str
    think_token_len: int
    tool_result_token_len: int
    parents: dict[int, float] = field(default_factory=dict)

    @property
    def token_len(self) -> int:
        return self.think_token_len + self.tool_result_token_len

    @property
    def full_text(self) -> str:
        if self.tool_result_text:
            return self.think_text + " " + self.tool_result_text
        return self.think_text

    @property
    def best_parent(self) -> Optional[int]:
        return next(iter(self.parents), None)

    @property
    def best_parent_score(self) -> float:
        return next(iter(self.parents.values()), 0.0)


class MemoryGraph:
    """
    Per-trajectory causal memory graph.

    Lifecycle:
        1. Created per trajectory in AgentData.__init__().
        2. add_node() called at end of each turn.
        3. get_context_turn_ids() called to decide which turns survive pruning.
        4. get_adjacency() called at rollout end to serialize for Trace to Learn.
    """

    def __init__(
        self,
        top_k: int = 2,
        scorer: Optional[CausalScorer] = None,
    ):
        self.top_k = top_k
        self.scorer: CausalScorer = scorer if scorer is not None else CrossEncoderScorer()
        self.nodes: list[MemoryNode] = []

    # ------------------------------------------------------------------
    # Build phase  (called during rollout, once per turn)
    # ------------------------------------------------------------------

    def add_node(
        self,
        think_text: str,
        tool_result_text: str,
        think_token_len: int,
        tool_result_token_len: int,
    ) -> MemoryNode:
        """Create a new node and compute parent edges via causal scoring."""
        turn_id = len(self.nodes)
        node = MemoryNode(
            turn_id=turn_id,
            think_text=think_text,
            tool_result_text=tool_result_text,
            think_token_len=think_token_len,
            tool_result_token_len=tool_result_token_len,
        )

        if turn_id > 0:
            prev_texts = [n.full_text for n in self.nodes]
            raw_scores = self.scorer.score(node.full_text, prev_texts)

            scored = sorted(
                zip(range(turn_id), raw_scores),
                key=lambda x: x[1],
                reverse=True,
            )
            node.parents = {pid: s for pid, s in scored[: self.top_k]}

        self.nodes.append(node)
        logger.debug(
            "MemoryGraph.add_node: turn_id=%d, think_tokens=%d, tool_tokens=%d, parents=%s",
            turn_id, think_token_len, tool_result_token_len, node.parents,
        )
        return node

    # ------------------------------------------------------------------
    # Prune to Act  (called during rollout)
    # ------------------------------------------------------------------

    def get_causal_chain(
        self,
        max_token_len: int,
        strategy: str = "greedy",
    ) -> list[int]:
        """
        Trace back through parent edges within the given token budget.

        Args:
            max_token_len: Token budget for the causal chain.
            strategy:      "greedy" (follow best_parent) or "optimal" (DFS).

        Returns:
            Chronologically ordered list of turn_ids.
        """
        if not self.nodes:
            return []

        latest = len(self.nodes) - 1

        if self.total_token_len() <= max_token_len:
            return list(range(len(self.nodes)))

        if strategy == "greedy":
            return self._causal_chain_greedy(latest, max_token_len)
        elif strategy == "optimal":
            return self._causal_chain_optimal(latest, max_token_len)
        else:
            raise ValueError(f"Unknown strategy: '{strategy}'. Choose 'greedy' or 'optimal'.")

    def _causal_chain_greedy(self, start: int, max_token_len: int) -> list[int]:
        chain: list[int] = []
        total = 0
        visited: set[int] = set()
        cur: Optional[int] = start

        while cur is not None and cur not in visited:
            node = self.nodes[cur]
            if total + node.token_len > max_token_len:
                break
            chain.append(cur)
            total += node.token_len
            visited.add(cur)
            cur = node.best_parent

        chain.reverse()
        return chain

    def _causal_chain_optimal(self, start: int, max_token_len: int) -> list[int]:
        best: list = [[], 0.0]

        def dfs(node_id: int, remaining: int, path: list[int], score_sum: float) -> None:
            node = self.nodes[node_id]
            if node.token_len > remaining:
                return
            path.append(node_id)
            if len(path) > len(best[0]) or (len(path) == len(best[0]) and score_sum > best[1]):
                best[0] = path.copy()
                best[1] = score_sum
            for parent_id, edge_score in node.parents.items():
                dfs(parent_id, remaining - node.token_len, path, score_sum + edge_score)
            path.pop()

        dfs(start, max_token_len, [], 0.0)
        best[0].reverse()
        return best[0]

    def get_context_turn_ids(
        self,
        context_max_tokens: int,
        short_memory_turns: int = 2,
        strategy: str = "greedy",
    ) -> list[int]:
        """
        Final set of turn_ids to retain, with overlap-aware budget:
            context = causal_chain ∪ recent(short_memory_turns)
            total token length of union ≤ context_max_tokens

        recent turns are always included first (short-term memory guarantee).
        The causal chain is given the full context_max_tokens budget so it can
        traverse through recent turns without penalising them twice. Only the
        causal-only turns (not already in recent) are checked against the
        remaining budget (context_max_tokens - recent_len). If they still
        exceed it, they are trimmed by causal score (highest score kept first).
        """
        recent = set(self.get_recent_turns(short_memory_turns))
        recent_len = sum(self.nodes[i].token_len for i in recent)

        # Give causal chain the full budget so recent turns inside the chain
        # are traversed freely (no double-counting penalty).
        causal = set(self.get_causal_chain(context_max_tokens, strategy=strategy))
        causal_only = causal - recent
        causal_only_len = sum(self.nodes[i].token_len for i in causal_only)

        if recent_len + causal_only_len <= context_max_tokens:
            return sorted(causal | recent)

        # causal-only turns exceed remaining budget — trim by causal score
        remaining = context_max_tokens - recent_len
        kept: list[int] = []
        used = 0
        for tid in sorted(causal_only, key=lambda i: self.nodes[i].best_parent_score, reverse=True):
            if used + self.nodes[tid].token_len <= remaining:
                kept.append(tid)
                used += self.nodes[tid].token_len
        return sorted(recent | set(kept))

    def get_recent_turns(self, n: int) -> list[int]:
        start = max(0, len(self.nodes) - n)
        return list(range(start, len(self.nodes)))

    def total_token_len(self) -> int:
        return sum(n.token_len for n in self.nodes)

    # ------------------------------------------------------------------
    # Trace to Learn  (called once at rollout end)
    # ------------------------------------------------------------------

    def get_adjacency(self) -> dict[str, Any]:
        """
        Serialize graph for downstream Trace to Learn training.

        Store in AgentLoopOutput.extra_fields["memory_graph"] ->
        DataProto.non_tensor_batch["memory_graph"].
        """
        return {
            "num_turns": len(self.nodes),
            "parents": {
                str(n.turn_id): {str(pid): s for pid, s in n.parents.items()}
                for n in self.nodes
            },
            "think_token_lens": {str(n.turn_id): n.think_token_len for n in self.nodes},
            "tool_result_token_lens": {str(n.turn_id): n.tool_result_token_len for n in self.nodes},
        }

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        lines = [f"MemoryGraph(turns={len(self.nodes)}, top_k={self.top_k})"]
        for n in self.nodes:
            parents_str = ", ".join(f"{pid}({s:.2f})" for pid, s in n.parents.items())
            lines.append(
                f"  [{n.turn_id}] think={n.think_token_len:4d} "
                f"tool={n.tool_result_token_len:4d} | parents=[{parents_str}]"
            )
        return "\n".join(lines)
