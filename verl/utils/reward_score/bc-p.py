import logging
import re
import string

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Lowercase, strip, remove articles / punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_finish_answer(solution_str: str) -> str | None:
    """Extract answer from finish tool call JSON."""
    pattern = r'"name"\s*:\s*"finish".*?"answer"\s*:\s*"(.*?)"(?:\s*[,}\]])'
    matches = re.findall(pattern, solution_str, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def _count_tool_calls(solution_str: str) -> dict:
    """Count different tool calls and extract search queries."""
    n_search = len(re.findall(r'"name"\s*:\s*"search"', solution_str))
    n_open_page = len(re.findall(r'"name"\s*:\s*"open_page"', solution_str))
    n_finish = len(re.findall(r'"name"\s*:\s*"finish"', solution_str))

    # Extract search queries from query_list
    queries = re.findall(r'"query_list"\s*:\s*\["([^"]+)"\]', solution_str)

    return {
        "n_search": n_search,
        "n_open_page": n_open_page,
        "n_finish": n_finish,
        "queries": queries,
    }


def _process_reward(solution_str: str) -> float:
    """Compute process reward based on behavioral signals.

    Rewards:
        +0.2  called finish (anti token-waste)
        +0.1  no repeated search queries (anti death-loop)
        +0.2  used open_page (encourage verification)
        Max process total: 0.5  (must stay below outcome reward 1.0)
    """
    reward = 0.0
    tools = _count_tool_calls(solution_str)

    # Called finish: +0.2
    if tools["n_finish"] > 0:
        reward += 0.2

    # No repeated queries: +0.2
    queries = tools["queries"]
    if queries and len(queries) == len(set(queries)):
        reward += 0.2

    # Used open_page: +0.1
    if tools["n_open_page"] > 0:
        reward += 0.1

    return reward


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # Process reward (always computed, provides cold-start signal)
    process = _process_reward(solution_str)

    # Outcome reward (binary, 0 or 1.0)
    outcome = 0.0
    predicted = _extract_finish_answer(solution_str)
    if predicted:
        expected = str(ground_truth).strip()
        if predicted.lower() == expected.lower():
            outcome = 1.0
        elif _normalize(predicted) == _normalize(expected):
            outcome = 1.0
        else:
            norm_pred = _normalize(predicted)
            norm_exp = _normalize(expected)
            if norm_exp and norm_pred:
                if norm_exp in norm_pred or norm_pred in norm_exp:
                    outcome = 1.0

    return {
        "score": process + outcome,  # training reward (process + outcome)
        "acc": outcome,              # pure accuracy (0 or 1) for validation
    }
