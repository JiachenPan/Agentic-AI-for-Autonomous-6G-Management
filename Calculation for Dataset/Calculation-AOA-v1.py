import re
import json
import csv
from math import pow
from typing import Dict, Any, List, Tuple, Optional

# ---------- Core allocation logic ----------

def allocate_spectrum(total_bw_mhz: float,
                      utilization: float,
                      p_min: float = 0.50,
                      p_max: float = 0.90,
                      gamma: float = 1.0,
                      step_mhz: int = 10,
                      rounding: str = "nearest") -> Dict[str, Any]:
    """
    Compute 5G NR-U / Wi-Fi 6 spectrum split from total bandwidth and utilization.

    p_NR = p_min + (p_max - p_min) * (1 - U)^gamma, then snap to step_mhz.
    utilization may be [0,1] or [0,100]; both are accepted.

    Returns: dict with nr_mhz, wifi_mhz, nr_pct, wifi_pct, p_nr_raw.
    """
    U = utilization / 100.0 if utilization > 1 else float(utilization)
    U = max(0.0, min(1.0, U))

    p_nr = p_min + (p_max - p_min) * pow(1 - U, gamma)
    p_nr = max(p_min, min(p_nr, p_max))

    nr_exact = total_bw_mhz * p_nr

    def align(x: float, step: int, mode: str = "nearest") -> int:
        if mode == "floor":
            return step * int(x // step)
        if mode == "ceil":
            return step * int(-(-x // step))  # integer ceil
        return step * int(round(x / step))

    nr_mhz = align(nr_exact, step_mhz, rounding)
    wifi_mhz = int(total_bw_mhz) - nr_mhz
    nr_pct = 100.0 * nr_mhz / total_bw_mhz
    wifi_pct = 100.0 - nr_pct

    return {
        "nr_mhz": nr_mhz,
        "wifi_mhz": wifi_mhz,
        "nr_pct": nr_pct,
        "wifi_pct": wifi_pct,
        "p_nr_raw": p_nr,
    }

# ---------- Text parsing helpers ----------

LOAD_CUES = {
    "very low": 0.15, "low": 0.25, "light": 0.30,
    "moderate": 0.50, "medium": 0.50,
    "high": 0.75, "heavy": 0.80, "very high": 0.90,
    "congested": 0.90, "busy": 0.80, "saturated": 0.95
}

def extract_total_bw_mhz(text: str) -> Optional[float]:
    # e.g., "Optimize 400MHz total spectrum", "total 400 MHz", "400 MHz spectrum"
    patt = re.compile(
        r'(?i)(?:total|optimi[sz]e|overall|combined|aggregate)[^\d]{0,20}'
        r'([0-9]+(?:\.[0-9]+)?)\s*mhz')
    m = patt.search(text)
    if not m:
        m = re.search(r'(?i)([0-9]+(?:\.[0-9]+)?)\s*mhz\s*(?:total|bandwidth|spectrum)', text)
    return float(m.group(1)) if m else None

def extract_utilization(text: str) -> Optional[float]:
    # numeric form
    m = re.search(r'(?i)channel\s*utilization\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%', text)
    if m:
        return float(m.group(1)) / 100.0
    # verbal load cues
    for cue, u in LOAD_CUES.items():
        if re.search(rf'(?i)\b{re.escape(cue)}\b\s*(?:load|traffic|utilization)?', text):
            return u
    return None

def parse_scene(text: str,
                default_bw_mhz: float = 400.0,
                default_U: float = 0.50) -> Tuple[float, float, Dict[str, Any]]:
    """
    Extract (total_bw_mhz, utilization in [0,1]) from a scene text.
    Falls back to defaults when missing.
    """
    bw = extract_total_bw_mhz(text)
    U = extract_utilization(text)

    info = {"bw_from_text": bw is not None, "util_from_text": U is not None}

    bw = bw if bw is not None else default_bw_mhz
    U = U if U is not None else default_U
    return float(bw), float(U), info

# ---------- Per-sample API ----------

def recommend_split_from_text(scene_text: str,
                              p_min: float = 0.50,
                              p_max: float = 0.90,
                              gamma: float = 1.0,
                              step_mhz: int = 10,
                              rounding: str = "nearest",
                              default_bw_mhz: float = 400.0,
                              default_U: float = 0.50) -> Dict[str, Any]:
    """
    One-shot: parse a single scene text and return a recommendation.
    """
    bw, U, parse_info = parse_scene(scene_text, default_bw_mhz, default_U)
    out = allocate_spectrum(bw, U, p_min, p_max, gamma, step_mhz, rounding)
    out.update({"total_bw_mhz": bw, "utilization": U, "parse_info": parse_info})
    return out

# ---------- Batch over your dataset ----------

def load_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Load either a JSON array file or a JSONL (one JSON object per line) file.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        raise ValueError("Top-level JSON is not a list.")
    except Exception:
        # Try JSONL
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

def get_user_text(item: Dict[str, Any]) -> Optional[str]:
    """
    Your dataset structure: item['conversations'] is a list of {from, value}.
    We pull the first message where from == 'user'.
    """
    conv = item.get("conversations", [])
    for m in conv:
        if m.get("from") == "user":
            return m.get("value", "")
    return None

def process_dataset(input_path: str,
                    output_csv_path: str,
                    defaults: Dict[str, Any] = None) -> None:
    """
    Iterate over all samples and write a CSV with recommended splits.
    Columns: id, nr_mhz, wifi_mhz, nr_pct, wifi_pct, total_bw_mhz, utilization, parsed_from_text.
    """
    if defaults is None:
        defaults = {"bw_mhz": 400.0, "U": 0.50}

    data = load_dataset(input_path)
    rows = []
    for item in data:
        uid = item.get("id", "")
        utext = get_user_text(item) or ""
        bw, U, parse_info = parse_scene(
            utext, default_bw_mhz=defaults["bw_mhz"], default_U=defaults["U"]
        )
        rec = allocate_spectrum(bw, U)
        rows.append({
            "id": uid,
            "nr_mhz": rec["nr_mhz"],
            "wifi_mhz": rec["wifi_mhz"],
            "nr_pct": round(rec["nr_pct"], 3),
            "wifi_pct": round(rec["wifi_pct"], 3),
            "total_bw_mhz": bw,
            "utilization_0_1": round(U, 5),
            "bw_from_text": parse_info["bw_from_text"],
            "util_from_text": parse_info["util_from_text"],
            "user_text": utext[:2000]  # optional for auditing
        })

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

# ---------- Example ----------

if __name__ == "__main__":
    # Single sample
    sample = ("Stadium network analysis - post-match period, cold weather: 39,775 active users detected through AoA measurements. "
              "Current spectrum allocation: 5G NR-U 215MHz, Wi-Fi 6 185MHz. Signal quality: poor (SNR: -3.4dB). "
              "Channel utilization: 20.8% (low load). AoA analysis shows 9 dominant directions with 7.7° spread, "
              "spatial diversity index: 5.1. Optimize 400MHz total spectrum.")
    print(recommend_split_from_text(sample))
    # -> {'nr_mhz': 330, 'wifi_mhz': 70, 'nr_pct': 82.5, 'wifi_pct': 17.5, ...}

    # Batch over your file (JSON or JSONL) and save a CSV
    # process_dataset("dataset.json", "split_recommendations.csv")
