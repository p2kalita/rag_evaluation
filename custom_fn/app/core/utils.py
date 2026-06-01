import json
import re

def _parse_json_response(raw: str) -> dict:
    raw = re.sub(r"```(?:json)?", "", raw).strip("`").strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)

    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))
