import re
from typing import Dict, List, Tuple, Optional

# --------- parser ---------
def parse_rssi_prompt(prompt: str) -> Dict[str, List[float]]:
    """
    Parse 'A_a: 320.0 ... D_f: 38.8' into {'A':[...], 'B':[...], ...}.
    Supports negative values (typical dBm) and arbitrary spacing.
    """
    pat = re.compile(r'\b([A-Za-z])_([A-Za-z])\s*:\s*(-?\d+(?:\.\d+)?)')
    sectors: Dict[str, List[float]] = {}
    for sec, sub, val in pat.findall(prompt):
        sectors.setdefault(sec.upper(), []).append(float(val))
    if not sectors:
        raise ValueError("No 'X_y: value' RSSI pairs found in the prompt.")
    return sectors

# --------- core allocator ---------
def allocate_from_rssi(
    prompt: str,
    total_bw_mhz_per_sector: float = 100.0,
    # thresholds: auto detects scale; override if you want fixed values
    threshold_mode: str = "auto",   # "auto" | "positive" | "dbm"
    low_thr_pos: float = 40.0,      # for positive-amplitude scale (0~400)
    high_thr_pos: float = 300.0,
    low_thr_dbm: float = -90.0,     # for dBm (negative)
    high_thr_dbm: float = -50.0,
    # scoring weights (tunable)
    w_low: float = 1.2,
    w_high: float = 0.6,
    w_mean: float = 0.3,
    # mapping range
    p_min: float = 0.37,
    p_max: float = 0.60,
    decimals: int = 1,
    verify_24: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Convert sector×sub-band RSSI to MHz allocation for 5G NR-U and Wi-Fi 6.

    - If verify_24=True, raises if total RSSI count != 24.
    - Auto thresholding:
        * positive scale (max > 100) -> use low_thr_pos/high_thr_pos
        * dBm-like (max <= 10) -> use low_thr_dbm/high_thr_dbm
    - Deterministic and robust: if all sector scores are identical,
      assigns the mid share (p_min+p_max)/2 to every sector.
    """
    sectors = parse_rssi_prompt(prompt)
    n_vals = sum(len(v) for v in sectors.values())
    if verify_24 and n_vals != 24:
        raise ValueError(f"Expected 24 RSSI values, got {n_vals}.")

    # flatten for global stats & auto threshold
    all_vals: List[float] = [x for arr in sectors.values() for x in arr]
    g_min, g_max = min(all_vals), max(all_vals)
    denom = max(g_max - g_min, 1e-9)

    if threshold_mode == "auto":
        if g_max > 100:   # amplitude-like (your screenshot)
            low_thr, high_thr = low_thr_pos, high_thr_pos
        else:             # typical dBm
            low_thr, high_thr = low_thr_dbm, high_thr_dbm
    elif threshold_mode == "positive":
        low_thr, high_thr = low_thr_pos, high_thr_pos
    else:  # "dbm"
        low_thr, high_thr = low_thr_dbm, high_thr_dbm

    # 1) sector scores
    scores: Dict[str, float] = {}
    for sec, arr in sectors.items():
        n_low = sum(1 for x in arr if x < low_thr)
        n_high = sum(1 for x in arr if x > high_thr)
        mean_norm = ((sum(arr) / len(arr)) - g_min) / denom
        score = w_low * n_low - w_high * n_high - w_mean * mean_norm
        scores[sec] = score

    s_min, s_max = min(scores.values()), max(scores.values())
    span = s_max - s_min

    # 2) score -> share
    def to_share(score: float) -> float:
        if span < 1e-12:
            return 0.5 * (p_min + p_max)  # identical scores -> equal middle share
        t = (score - s_min) / span
        return p_min + (p_max - p_min) * t

    # 3) per-sector allocations
    out: Dict[str, Dict[str, float]] = {}
    for sec, score in scores.items():
        p_nr = max(p_min, min(p_max, to_share(score)))
        nr_mhz = round(total_bw_mhz_per_sector * p_nr, decimals)
        wifi_mhz = round(total_bw_mhz_per_sector - nr_mhz, decimals)
        out[sec] = {
            "NR-U": nr_mhz,
            "WiFi6": wifi_mhz,
            "NR-U%": round(100.0 * p_nr, decimals),
            "WiFi6%": round(100.0 - 100.0 * p_nr, decimals),
        }

    # (optional) attach debug info for auditing
    out["_meta"] = {
        "count_values": n_vals,
        "thresholds": {"low": low_thr, "high": high_thr, "mode": threshold_mode},
        "scores": scores
    }
    return out

# --------- quick demo ---------
if __name__ == "__main__":
    demo = ("A_a: 320.0 A_b: 324.2 A_c: 299.4 A_d: 283.0 A_e: 31.1 A_f: 20.7 "
            "B_a: 329.6 B_b: 39.8 B_c: 221.6 B_d: 289.9 B_e: 238.9 B_f: 9.2 "
            "C_a: 352.8 C_b: 204.3 C_c: 246.1 C_d: 154.0 C_e: 211.0 C_f: 5.4 "
            "D_a: 316.2 D_b: 206.2 D_c: 162.3 D_d: 0.5 D_e: 8.6 D_f: 38.8")
    print(allocate_from_rssi(demo, verify_24=True))
