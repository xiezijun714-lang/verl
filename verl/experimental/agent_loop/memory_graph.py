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
Directed edges represent causal/logical dependency: node t → parents(t) means
"turn t's reasoning is built upon the observations of its parent turns".

Each node stores only:
  - parents: dict[turn_id -> score], sorted descending by score.
             Sufficient for both Prune to Act (backward causal chain retrieval)
             and Trace to Learn (forward advantage computation iterates t=0..T,
             and for each t reads parents[t] which are all already computed).

Similarity backend is pluggable via the SimilarityEncoder protocol:
  - JaccardEncoder:      zero latency, zero dependencies (default / fallback)
  - EmbeddingEncoder:    bi-encoder via sentence-transformers, ~1-3ms CPU (recommended)
  - CrossEncoderEncoder: joint encoding, best logical-dependency quality, ~5-10ms CPU per pair
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Similarity encoder protocol — plug in any backend
# ---------------------------------------------------------------------------

@runtime_checkable
class SimilarityEncoder(Protocol):
    """Any object implementing score() can be used as the similarity backend."""

    def score(self, query: str, candidates: list[str]) -> list[float]:
        """
        Return a similarity score for each (query, candidate) pair.
        Higher score = more causally related.
        len(return) == len(candidates).
        """
        ...


class JaccardEncoder:
    """
    Token-level Jaccard similarity.
    Zero latency, zero dependencies. Poor at semantic / logical relationships.
    Use only for unit tests or as a quick smoke-test fallback.
    """

    def score(self, query: str, candidates: list[str]) -> list[float]:
        q_tokens = set(query.lower().split())
        results = []
        for c in candidates:
            c_tokens = set(c.lower().split())
            union = q_tokens | c_tokens
            results.append(len(q_tokens & c_tokens) / len(union) if union else 0.0)
        return results


class EmbeddingEncoder:
    """
    Bi-encoder using sentence-transformers (dual-tower cosine similarity).

    Good at capturing topical / semantic overlap between turns.
    Weaker for strict logical dependency (e.g. "turn 3 uses the file found in turn 1").

    Recommended model : 'all-MiniLM-L6-v2'  (~22 MB, CPU ~1-3 ms per call)
    Chinese-friendly  : 'BAAI/bge-small-zh-v1.5' (~24 MB)

    Install: pip install sentence-transformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        try:
            from sentence_transformers import SentenceTransformer, util as st_util
            self._model = SentenceTransformer(model_name, device=device)
            self._util = st_util
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for EmbeddingEncoder. "
                "Install with: pip install sentence-transformers"
            )

    def score(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        embeddings = self._model.encode(
            [query] + candidates,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        q_emb, c_embs = embeddings[0], embeddings[1:]
        return self._util.cos_sim(q_emb, c_embs)[0].tolist()


class CrossEncoderEncoder:
    """
    Cross-encoder: jointly encodes query+candidate pairs.

    Better at capturing logical dependency than bi-encoder because it sees
    both texts simultaneously. Tradeoff: O(n) forward passes vs O(1) for bi-encoder.

    Recommended model : 'cross-encoder/ms-marco-MiniLM-L-6-v2'  (~22 MB, CPU ~5-10 ms per pair)

    Install: pip install sentence-transformers
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ):
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_name, device=device)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderEncoder. "
                "Install with: pip install sentence-transformers"
            )

    def score(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        pairs = [[query, c] for c in candidates]
        return self._model.predict(pairs).tolist()


def build_encoder(
    encoder_type: str = "jaccard",
    model_name: Optional[str] = None,
    device: str = "cpu",
) -> SimilarityEncoder:
    """
    Factory for similarity encoders.

    Args:
        encoder_type: "jaccard" | "embedding" | "cross_encoder"
        model_name:   Override default model name (optional).
        device:       "cpu" or "cuda".

    Example (in ToolAgentLoop.__init__):
        self._memory_encoder = build_encoder("embedding", device="cpu")
        # Then pass to AgentData:
        agent_data = AgentData(..., memory_encoder=self._memory_encoder)
    """
    if encoder_type == "jaccard":
        return JaccardEncoder()
    elif encoder_type == "embedding":
        return EmbeddingEncoder(model_name=model_name or "all-MiniLM-L6-v2", device=device)
    elif encoder_type == "cross_encoder":
        return CrossEncoderEncoder(
            model_name=model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=device,
        )
    else:
        raise ValueError(
            f"Unknown encoder_type: '{encoder_type}'. "
            "Choose from: 'jaccard', 'embedding', 'cross_encoder'."
        )


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class MemoryNode:
    """
    One node = one complete assistant turn (think + tool_result).

    Attributes:
        turn_id:              Zero-indexed position in this trajectory.
        think_text:           Model's chain-of-thought / reasoning for this turn.
        tool_result_text:     Tool response observed after this turn (empty string for the
                              final turn that produces no tool call).
        think_token_len:      Tokens occupied by think_text.
        tool_result_token_len:Tokens occupied by tool_result_text (0 for final turn).
        parents:              {prev_turn_id -> similarity_score}, sorted descending by score.
                              Encodes "this turn's logic depends on these prior turns".
                              Used for:
                                - Prune to Act: backward causal chain retrieval via best_parent.
                                - Trace to Learn: A_t formula reads parents[t] in a forward pass.
    """

    turn_id: int
    think_text: str
    tool_result_text: str
    think_token_len: int
    tool_result_token_len: int
    parents: dict[int, float] = field(default_factory=dict)

    @property
    def token_len(self) -> int:
        """Total tokens this turn occupies (think + tool_result)."""
        return self.think_token_len + self.tool_result_token_len

    @property
    def full_text(self) -> str:
        """Concatenated think + tool_result. Used as default similarity input."""
        if self.tool_result_text:
            return self.think_text + " " + self.tool_result_text
        return self.think_text

    @property
    def best_parent(self) -> Optional[int]:
        """turn_id of the most causally related predecessor, or None for root node."""
        return next(iter(self.parents), None)

    @property
    def best_parent_score(self) -> float:
        """Similarity score of best_parent, or 0.0 for root node."""
        return next(iter(self.parents.values()), 0.0)


class MemoryGraph:
    """
    Per-trajectory causal memory graph.

    Lifecycle:
        1. One MemoryGraph is created per trajectory inside AgentData.__init__().
        2. add_node() is called at the end of each turn (after tool_result is appended).
        3. get_context_turn_ids() is called at the start of each _handle_generating_state()
           to decide which turns survive context pruning.
        4. get_adjacency() is called once at rollout end to serialize graph structure
           into AgentLoopOutput.extra_fields["memory_graph"], which flows automatically
           into DataProto.non_tensor_batch["memory_graph"] for Trace to Learn.

    Thread safety: NOT thread-safe. One instance per trajectory.
    """

    def __init__(
        self,
        top_k: int = 2,
        encoder: Optional[SimilarityEncoder] = None,
    ):
        """
        Args:
            top_k:   Maximum number of parent edges per node (k in the paper).
            encoder: Similarity backend. Defaults to JaccardEncoder if None.
                     Pass an EmbeddingEncoder or CrossEncoderEncoder for real use.
        """
        self.top_k = top_k
        self.encoder: SimilarityEncoder = encoder if encoder is not None else JaccardEncoder()
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
        """
        Create a new node for the just-completed turn and compute parent edges.

        Call this AFTER tool_result tokens have been appended to the context,
        i.e., at the end of _handle_processing_tools_state() or
        _handle_interacting_state(), just before returning AgentState.GENERATING.

        Args:
            think_text:            Model's reasoning text for this turn.
            tool_result_text:      Tool response for this turn (empty string if final turn).
            think_token_len:       Token count of think_text.
            tool_result_token_len: Token count of tool_result_text (0 if final turn).

        Returns:
            The newly created MemoryNode.
        """
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
            raw_scores = self.encoder.score(node.full_text, prev_texts)

            # Sort descending, keep top-k
            scored = sorted(
                zip(range(turn_id), raw_scores),
                key=lambda x: x[1],
                reverse=True,
            )
            node.parents = {pid: s for pid, s in scored[: self.top_k]}

        self.nodes.append(node)
        logger.debug(
            "MemoryGraph.add_node: turn_id=%d, token_len=%d, parents=%s",
            turn_id,
            token_len,
            node.parents,
        )
        return node

    # ------------------------------------------------------------------
    # Prune to Act queries  (called during rollout)
    # ------------------------------------------------------------------

    def get_causal_chain(
        self,
        max_token_len: int,
        strategy: str = "greedy",
    ) -> list[int]:
        """
        Select a causal chain starting from the latest node, tracing back
        through parent edges, within the given token budget.

        Args:
            max_token_len: Token budget for the causal chain.
            strategy:
                "greedy"  — Always follow best_parent. O(n). Fast, good default.
                            May miss shorter alternative paths when budget is tight.
                "optimal" — DFS over all parent edges. Finds the path that
                            maximises (1) number of nodes, then (2) sum of edge
                            scores as a tiebreaker. Practical for trajectories
                            up to ~30 turns with top_k=2 due to budget pruning.

        Returns:
            Chronologically ordered list of turn_ids (oldest → newest).
        """
        if not self.nodes:
            return []

        latest = len(self.nodes) - 1

        # Fast path: everything fits — no need to search
        if self.total_token_len() <= max_token_len:
            return list(range(len(self.nodes)))

        if strategy == "greedy":
            return self._causal_chain_greedy(latest, max_token_len)
        elif strategy == "optimal":
            return self._causal_chain_optimal(latest, max_token_len)
        else:
            raise ValueError(
                f"Unknown strategy: '{strategy}'. Choose 'greedy' or 'optimal'."
            )

    def _causal_chain_greedy(self, start: int, max_token_len: int) -> list[int]:
        """Follow best_parent greedily until budget is exhausted."""
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
        """
        DFS over all parent edges.

        Optimisation objective (lexicographic):
          1. Maximise number of nodes in the chain  (coverage)
          2. Maximise sum of traversed edge scores  (causal confidence)

        Budget pruning makes this practical for typical agentic trajectories
        (≤ 30 turns, top_k ≤ 2). Worst-case O(top_k ^ n) but rarely reached.
        """
        # best[0]: best path found so far (list of turn_ids, reverse-chronological)
        # best[1]: sum of edge scores for that path
        best: list = [[], 0.0]

        def dfs(node_id: int, remaining: int, path: list[int], score_sum: float) -> None:
            node = self.nodes[node_id]
            if node.token_len > remaining:
                return

            path.append(node_id)

            # Update best: more nodes wins; equal nodes → higher score sum wins
            if len(path) > len(best[0]) or (
                len(path) == len(best[0]) and score_sum > best[1]
            ):
                best[0] = path.copy()
                best[1] = score_sum

            for parent_id, edge_score in node.parents.items():
                dfs(
                    parent_id,
                    remaining - node.token_len,
                    path,
                    score_sum + edge_score,
                )

            path.pop()

        dfs(start, max_token_len, [], 0.0)
        best[0].reverse()  # chronological order
        return best[0]

    def get_recent_turns(self, n: int) -> list[int]:
        """
        Return the turn_ids of the n most recent turns, in chronological order.
        These form the 'short memory' that is always preserved regardless of causality.
        """
        start = max(0, len(self.nodes) - n)
        return list(range(start, len(self.nodes)))

    def get_context_turn_ids(
        self,
        causal_max_tokens: int,
        short_memory_turns: int = 2,
        strategy: str = "greedy",
    ) -> list[int]:
        """
        Compute the final set of turn_ids to retain after pruning.

        Result = union(causal_chain, recent_turns), deduplicated and sorted.

        As described in the ECHO paper:
            causal_max_length = max_length - short_memory_length
            context = causal_chain(causal_max_length) ∪ recent(short_memory_turns)

        Args:
            causal_max_tokens:  Token budget for the causal chain.
            short_memory_turns: Number of most-recent turns always included.
            strategy:           Passed through to get_causal_chain().
                                "greedy" (default) or "optimal".

        Returns:
            Sorted list of turn_ids to keep in the pruned context.
        """
        causal = set(self.get_causal_chain(causal_max_tokens, strategy=strategy))
        recent = set(self.get_recent_turns(short_memory_turns))
        return sorted(causal | recent)

    def total_token_len(self) -> int:
        """Sum of token_len across all recorded turns."""
        return sum(n.token_len for n in self.nodes)

    # ------------------------------------------------------------------
    # Trace to Learn queries  (called once at rollout end)
    # ------------------------------------------------------------------

    def get_adjacency(self) -> dict[str, Any]:
        """
        Serialize the full graph structure for downstream Trace to Learn.

        Store this in AgentLoopOutput.extra_fields["memory_graph"].
        It will automatically propagate to DataProto.non_tensor_batch["memory_graph"]
        via the existing extra_fields collection in _postprocess().

        Schema:
            {
                "num_turns": int,
                "parents":   {str(turn_id): {str(parent_id): score, ...}, ...},
                "token_lens":{str(turn_id): token_len, ...},
            }

        Note: JSON / numpy object-array keys must be strings.
        """
        return {
            "num_turns": len(self.nodes),
            "parents": {
                str(n.turn_id): {str(pid): s for pid, s in n.parents.items()}
                for n in self.nodes
            },
            "think_token_lens": {
                str(n.turn_id): n.think_token_len
                for n in self.nodes
            },
            "tool_result_token_lens": {
                str(n.turn_id): n.tool_result_token_len
                for n in self.nodes
            },
        }

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

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
