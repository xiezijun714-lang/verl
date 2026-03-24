import re

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # 提取最后一个 <answer>...</answer>
    matches = re.findall(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if not matches:
        return 0.0
    predicted = matches[-1].strip().lower()
    expected = str(ground_truth).strip().lower()
    return 1.0 if predicted == expected else 0.0
