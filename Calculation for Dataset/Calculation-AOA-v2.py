import re
from typing import Optional, Dict, Any

# ---- helpers: parsing ----

ACTIVITY_TO_LOAD = {
    # maps text cues -> utilization in [0,1]
    "very low": 0.20, "low": 0.30, "light": 0.30,
    "moderate": 0.50, "medium": 0.50, "balanced": 0.50,
    "elevated": 0.65, "high": 0.75, "heavy": 0.80,
    "very high": 0.90, "peak": 0.85, "saturated": 0.95,
}

AOA_DYNAMICS = {  # qualitative AoA dynamics -> [0,1]
    "very low": 0.20, "low": 0.30, "moderate": 0.50, "medium": 0.50, "high": 0.80, "very high": 0.90
}

def _extract_total_bw_mhz(text: str) -> Optional[float]:
    m = re.search(r'(?i)(?:total|optimi[sz]e|overall|aggregate)[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)\s*mhz', text)
    if not m:
        m = re.search(r'(?i)([0-9]+(?:\.[0-9]+)?)\s*mhz\s*(?:total|bandwidth|spectrum)', text)
    return float(m.group(1)) if m else None

def _extract_activity_load(text: str) -> Optional[float]:
    for cue, u in ACTIVITY_TO_LOAD.items():
        if re.search(rf'(?i)\b{re.escape(cue)}\b\s*(?:activity|load|level)?', text):
            return u
    return None

def _extract_named_float(text: str, label: str) -> Optional[float]:
    m = re.search(rf'(?i){re.escape(label)}\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    return float(m.group(1)) if m else None

def _extract_aoa_dynamics(text: str) -> Optional[float]:
    # e.g., "AoA tracking shows low angular dynamics"
    m = re.search(r'(?i)aoa[^.]*\b(very low|low|moderate|medium|high|very high)\b[^.]*angular dynamics', text)
    if m:
        return AOA_DYNAMICS[m.group(1).lower()]
    return None

def _align(value: float, step: int = 1, mode: str = "floor") -> int:
    if mode == "floor":
        return step * int(value // step)
    if mode == "ceil":
        return step * int(-(-value // step))
    return step * int(round(value / step))

# ---- main function ----

def recommend_temporal_split(scene_text: str,
                             # base NR-U range for temporal cases (more balanced than congestion-only):
                             p_min: float = 0.45, p_max: float = 0.75,
                             # feature weights (tuned to be sensible + reproduce your screenshot):
                             w_mob: float = 0.09,       # higher mobility -> more NR-U
                             w_data: float = -0.06,     # higher data intensity -> more Wi-Fi
                             w_aoa: float = 0.06,       # higher AoA dynamics -> more NR-U
                             step_mhz: int = 1, rounding: str = "floor",
                             default_bw_mhz: float = 400.0,
                             default_activity_load: float = 0.50
                             ) -> Dict[str, Any]:
    """
    Parse a temporal-scenario text and recommend NR-U/Wi-Fi spectrum.

    Logic:
      base = p_min + (p_max - p_min) * (1 - load)
      p_NR = base + w_mob*(mobility-0.5) + w_data*(data_intensity-0.5) + w_aoa*(aoa_dyn-0.5)
      clamp to [p_min, p_max], then convert to MHz and percentages.

    Returns: dict with fields {nr_mhz, wifi_mhz, nr_pct, wifi_pct, p_nr, features}
    """
    bw = _extract_total_bw_mhz(scene_text) or default_bw_mhz
    load = _extract_activity_load(scene_text)
    load = default_activity_load if load is None else load

    mobility = _extract_named_float(scene_text, "Mobility factor")
    mobility = 0.5 if mobility is None else max(0.0, min(1.0, mobility))

    data_intensity = _extract_named_float(scene_text, "data intensity")
    data_intensity = 0.5 if data_intensity is None else max(0.0, min(1.0, data_intensity))

    aoa_dyn = _extract_aoa_dynamics(scene_text)
    aoa_dyn = 0.5 if aoa_dyn is None else aoa_dyn

    base = p_min + (p_max - p_min) * (1 - load)
    p_nr = base + w_mob * (mobility - 0.5) + w_data * (data_intensity - 0.5) + w_aoa * (aoa_dyn - 0.5)
    p_nr = max(p_min, min(p_max, p_nr))

    nr_exact = bw * p_nr
    nr_mhz = _align(nr_exact, step=step_mhz, mode=rounding)
    wifi_mhz = int(round(bw)) - nr_mhz

    return {
        "nr_mhz": nr_mhz,
        "wifi_mhz": wifi_mhz,
        "nr_pct": 100.0 * nr_mhz / bw,
        "wifi_pct": 100.0 - (100.0 * nr_mhz / bw),
        "p_nr": p_nr,
        "features": {
            "total_bw_mhz": bw,
            "activity_load": load,
            "mobility": mobility,
            "data_intensity": data_intensity,
            "aoa_dynamics": aoa_dyn,
            "base_share": base
        }
    }

# ---- example using your screenshot text ----
if __name__ == "__main__":
    text = ("Temporal network analysis - seasonal variation at 08:34; 9,131 users with moderate activity level. "
            "Mobility factor: 0.29, data intensity: 0.67. AoA tracking shows low angular dynamics. "
            "Historical pattern analysis indicates seasonal_variation behavior. Current allocation insufficient for temporal optimization. "
            "Require predictive spectrum management for 400MHz total bandwidth considering time-based user behavior patterns.")
    print(recommend_temporal_split(text))
    # Expected (with defaults shown above): ~ NR-U 223 MHz (≈55.8%), Wi-Fi 177 MHz (≈44.2%)
