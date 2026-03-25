import json
import logging
import os
import re
import string
from urllib import request, error

logger = logging.getLogger(__name__)

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
JUDGE_MODEL = "qwen-turbo"  # 便宜快速，够用于判断语义等价
JUDGE_TIMEOUT = 15  # seconds

JUDGE_PROMPT = """\
You are a strict answer equivalence judge.
Determine whether the predicted answer is semantically equivalent to the expected answer.
They may differ in formatting, punctuation, articles (a/an/the), abbreviations, or minor wording, but must refer to the same entity/concept.

Expected answer: {expected}
Predicted answer: {predicted}

Reply with ONLY one word: "YES" or "NO"."""


def _normalize(text: str) -> str:
    """Lowercase, strip, remove articles / punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _llm_judge(predicted: str, expected: str) -> float:
    """Call DashScope API to judge semantic equivalence. Returns 1.0 / 0.0 / -1.0 (error)."""
    if not DASHSCOPE_API_KEY:
        return -1.0

    payload = json.dumps({
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "user", "content": JUDGE_PROMPT.format(expected=expected, predicted=predicted)}
        ],
        "max_tokens": 8,
        "temperature": 0.0,
    }).encode("utf-8")

    req = request.Request(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        },
        method="POST",
    )

    try:
        proxy_url = os.environ.get("https_proxy") or os.environ.get("http_proxy", "")
        if proxy_url:
            proxy_handler = request.ProxyHandler({"https": proxy_url, "http": proxy_url})
            opener = request.build_opener(proxy_handler)
        else:
            opener = request.build_opener()
        with opener.open(req, timeout=JUDGE_TIMEOUT) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            answer = result["choices"][0]["message"]["content"].strip().upper()
            if "YES" in answer:
                return 1.0
            return 0.0
    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        return -1.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    matches = re.findall(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if not matches:
        return 0.0

    predicted = matches[-1].strip()
    expected = str(ground_truth).strip()

    # 1) Exact match (case-insensitive) — fast path, no API call
    if predicted.lower() == expected.lower():
        return 1.0

    # 2) Normalized match (remove articles, punctuation, extra whitespace)
    if _normalize(predicted) == _normalize(expected):
        return 1.0

    # 3) LLM judge for semantic equivalence
    llm_result = _llm_judge(predicted, expected)
    if llm_result >= 0:
        return llm_result

    # 4) Fallback: rule-based (only if API failed)
    norm_pred = _normalize(predicted)
    norm_exp = _normalize(expected)
    if norm_exp and norm_pred:
        if norm_exp in norm_pred or norm_pred in norm_exp:
            return 0.5

    return 0.0
