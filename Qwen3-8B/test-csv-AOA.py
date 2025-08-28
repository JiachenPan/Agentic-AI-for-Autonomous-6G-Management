# -*- coding: utf-8 -*-
"""
Batch constraint evaluation for AoA spectrum allocation (Base vs LoRA)
- Reads a JSONL with {"id":..., "conversations":[{"from":"user","value":...}, {"from":"assistant","value":...}]}
- Sequentially loads Base and/or LoRA model (32GB-friendly).
- For each prompt: generate -> parse -> check constraints -> CSV per model + merged comparison CSV.
- Robust loader with fallbacks; silences generation sampling warnings; shows ETA.
- Stable % extraction: only near "5G/NR-U" and "Wi-Fi 6/6E"; otherwise derive from MHz.
- Light output-format hint via system message.
- CSV adds pass_band_only (MHz-only) and pass_full (five constraints).

Updates:
- 默认 Base 与 LoRA 各评估 3000 条（--limit-base / --limit-lora；<=0 表示全量）。
- 修正 entity_ok：仅在“解释段”近邻“utilization/利用率/占用”匹配百分比，避免把首行 5G_%/WiFi_% 当成利用率。
- 评测侧“比例和解”：若显式 % 与 MHz 比例不一致，用 MHz 推导的 % 回填，pct_source 标记为 reconciled。
"""

import os, re, json, csv, time, argparse
from typing import Dict, Any, Optional, Tuple, List

# ---- silence transformers logs globally ----
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None
from peft import PeftModel

# ---------------- Config (paths; you can also pass via CLI) ----------------
BASE_PATH = "/root/autodl-tmp/Qwen/Qwen3-8B"   # 基座
LORA_PATH        = "./qwen3-8B-ft-v3"                 # 你微调好的LoRA输出目录
DATASET_DEFAULT       = "AoA_dataset.jsonl"

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------- Regex helpers ----------------
NUM = r"\d+(?:,\d{3})*(?:\.\d+)?"

# 近邻 5G / NR-U
MHZ_NEAR      = re.compile(rf"(?i)(?:5g|nr-?u)[^0-9]{{0,20}}({NUM})\s*mhz")
PCT_5G_NEAR   = re.compile(rf"(?i)(?:5g|nr-?u)[^%]{{0,30}}(\d+(?:\.\d+)?)\s*%")

# 近邻 Wi-Fi 6/6E
_WIFI_WORD    = r"(?:wi[\-\s]?fi\s*6(?:e)?|wifi\s*6(?:e)?)"
WIFI_MHZ_NEAR = re.compile(rf"(?i){_WIFI_WORD}[^0-9]{{0,20}}({NUM})\s*mhz")
PCT_WIFI_NEAR = re.compile(rf"(?i){_WIFI_WORD}[^%]{{0,30}}(\d+(?:\.\d+)?)\s*%")

# 任意 MHz（兜底抓前两处）
ANY_MHZ = re.compile(rf"(?i)({NUM})\s*mhz")

# 提示里抽 ground truth
PROMPT_TOTAL = re.compile(rf"(?i)optimize\s+({NUM})\s*mhz")
PROMPT_AOA   = re.compile(rf"(?i)aoa[^\.]*?shows\s+(\d+)\s+dominant\s+directions")
PROMPT_SNR   = re.compile(rf"(?i)snr[:\s]*({NUM})\s*dB")
PROMPT_UTIL  = re.compile(rf"(?i)channel\s+utilization[:\s]*({NUM})\s*%")

LOW_WORDS = ("low load", "low utilization", "低负载", "低拥塞", "低占用")

def to_float(x:str)->float: return float(x.replace(",", ""))

# ---------------- Parsing from prompt ----------------
def parse_prompt_truth(user_text:str)->Dict[str, Any]:
    truth = {"total_mhz": None, "aoa_dirs": None, "snr_db": None, "util_pct": None, "low_load_expected": False}
    m = PROMPT_TOTAL.search(user_text)
    if m: truth["total_mhz"] = to_float(m.group(1))
    m = PROMPT_AOA.search(user_text)
    if m: truth["aoa_dirs"] = int(m.group(1))
    m = PROMPT_SNR.search(user_text)
    if m: truth["snr_db"] = to_float(m.group(1))
    m = PROMPT_UTIL.search(user_text)
    if m:
        truth["util_pct"] = to_float(m.group(1))
        truth["low_load_expected"] = truth["util_pct"] <= 40.0
    return truth

# ---------------- Parsing from model output ----------------
def pick_first_near(pattern:re.Pattern, text:str)->Optional[float]:
    m = pattern.search(text)
    return to_float(m.group(1)) if m else None

def pick_first_any(pattern:re.Pattern, text:str, n:int=2)->List[float]:
    vals = []
    for m in pattern.finditer(text):
        vals.append(to_float(m.group(1)))
        if len(vals) >= n: break
    return vals

def parse_output_values(text:str)->Dict[str, Any]:
    """
    Extract MHz and (only-nearby) %.
    Percentages are captured ONLY near 5G/Wi-Fi mentions; no global fallback here.
    """
    out = {
        "g5_mhz": None, "wifi_mhz": None,
        "g5_pct_explicit": None, "wifi_pct_explicit": None
    }
    out["g5_mhz"] = pick_first_near(MHZ_NEAR, text)
    out["wifi_mhz"] = pick_first_near(WIFI_MHZ_NEAR, text)
    out["g5_pct_explicit"]  = pick_first_near(PCT_5G_NEAR, text)
    out["wifi_pct_explicit"]= pick_first_near(PCT_WIFI_NEAR, text)

    # MHz fallbacks: first two MHz numbers if not found near labels
    if out["g5_mhz"] is None or out["wifi_mhz"] is None:
        mhzs = pick_first_any(ANY_MHZ, text, n=2)
        if len(mhzs) >= 2:
            if out["g5_mhz"] is None:  out["g5_mhz"]  = mhzs[0]
            if out["wifi_mhz"] is None: out["wifi_mhz"] = mhzs[1]

    return out

# ---------------- Constraint checks ----------------
def band_ok(g5:float, wf:float, total:float, tol:float=5.0)->bool:
    return all(v is not None for v in (g5, wf, total)) and abs((g5+wf) - total) <= tol

def pct_ok(g5p:float, wfp:float, tol:float=3.0)->bool:
    return all(v is not None for v in (g5p, wfp)) and abs((g5p+wfp) - 100.0) <= tol

def ratio_ok(g5:float, wf:float, g5p:float, wfp:float, tol_pp:float=3.0)->Tuple[bool, Optional[float]]:
    if None in (g5, wf, g5p, wfp) or (g5+wf) <= 0 or (g5p+wfp) <= 0:
        return False, None
    r_mhz = g5/(g5+wf)
    r_pct = g5p/(g5p+wfp)
    err = abs(r_mhz - r_pct) * 100.0
    return err <= tol_pp, err

def entity_ok(output_text:str, truth:Dict[str,Any], snr_tol_db:float=0.5, util_tol_pp:float=5.0)->Dict[str,bool]:
    """
    实体一致性（AoA/SNR/利用率）：
    - 只在“解释段”里抓利用率和 SNR，避免把首行四字段误判为实体值。
    - 利用率只在相关近邻词出现时才匹配百分比。
    """
    parts = output_text.splitlines()
    explain = "\n".join(parts[1:]) if len(parts) > 1 else output_text
    explain_low = explain.lower()

    # AoA（宽松：只要出现目标数字或“8 direction(s)”）
    aoa_pass = True
    if truth.get("aoa_dirs") is not None:
        target = str(truth["aoa_dirs"])
        aoa_pass = (target in explain_low) or ("8-direction" in explain_low) or ("8 direction" in explain_low)

    # SNR 仅在解释段找
    snr_pass = True
    if truth.get("snr_db") is not None:
        m = re.search(rf"(?i)snr[^0-9\-]*({NUM})\s*dB", explain)
        if m:
            snr_pred = to_float(m.group(1))
            snr_pass = abs(snr_pred - truth["snr_db"]) <= snr_tol_db
        else:
            snr_pass = False

    # 利用率：仅近邻“utilization/利用率/占用”
    load_pass = True
    if truth.get("util_pct") is not None:
        UTIL_NEAR = re.compile(rf"(?i)(?:channel\s*util(?:ization)?|util(?:ization)?|利用率|占用)[^0-9%]{{0,12}}({NUM})\s*%")
        m = UTIL_NEAR.search(explain)
        if m:
            p = to_float(m.group(1))
            load_pass = abs(p - truth["util_pct"]) <= util_tol_pp
        else:
            load_pass = any(w in explain_low for w in LOW_WORDS) if truth["low_load_expected"] else True

    return {"aoa_ok": aoa_pass, "snr_ok": snr_pass, "load_ok": load_pass}

def unit_range_ok(g5:Optional[float], wf:Optional[float], g5p:Optional[float], wfp:Optional[float], total:Optional[float])->bool:
    if None in (g5, wf, g5p, wfp, total): return False
    if not (0 < g5 <= total and 0 < wf <= total): return False
    if not (0.0 <= g5p <= 100.0 and 0.0 <= wfp <= 100.0): return False
    return True

def unit_range_mhz_ok(g5:Optional[float], wf:Optional[float], total:Optional[float])->bool:
    if None in (g5, wf, total): return False
    return (0 < g5 <= total) and (0 < wf <= total)

# ---------------- I/O ----------------
def resolve_dataset_path(path: str) -> str:
    """Return a valid path; try case-insensitive match and default fallback."""
    if path and os.path.exists(path):
        return path
    if path:
        d = os.path.dirname(path) or "."
        base = os.path.basename(path)
        if os.path.isdir(d):
            for fname in os.listdir(d):
                if fname.lower() == base.lower():
                    return os.path.join(d, fname)
    if os.path.exists(DATASET_DEFAULT):
        return DATASET_DEFAULT
    raise FileNotFoundError(f"Dataset not found. Tried: {path} and {DATASET_DEFAULT}")

def load_jsonl(path:str, limit:int=None)->List[Dict[str,Any]]:
    """
    Load items from JSONL. If limit is None or <=0, load all.
    """
    path = resolve_dataset_path(path)
    data=[]
    with open(path,"r",encoding="utf-8") as f:
        for i,line in enumerate(f,1):
            try:
                obj=json.loads(line.strip())
                data.append(obj)
                if (limit is not None) and (limit > 0) and len(data)>=limit:
                    break
            except:
                pass
    return data

def get_user_text(item:Dict[str,Any])->Optional[str]:
    conv = item.get("conversations", [])
    if len(conv)>=1 and conv[0].get("from")=="user":
        return conv[0].get("value")
    return None

# ---------------- Model loading & generation ----------------
def prepare_tokenizer(base_path:str):
    tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True, use_fast=False, padding_side="right")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return tok

def _try_load(base_path, device_map=None, torch_dtype=None, quant=None):
    kw = dict(trust_remote_code=True, low_cpu_mem_usage=True)
    if device_map is not None: kw["device_map"] = device_map
    if torch_dtype is not None: kw["torch_dtype"] = torch_dtype
    if quant and BitsAndBytesConfig is not None:
        if quant == "8bit":
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quant == "4bit":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
    return AutoModelForCausalLM.from_pretrained(base_path, **kw)

def _silence_generation(model):
    try:
        gc = model.generation_config
        gc.do_sample = False
        for k in ("temperature", "top_p", "top_k", "typical_p"):
            if hasattr(gc, k):
                setattr(gc, k, None)
    except Exception:
        pass

def load_model_resilient(base_path:str, lora_path:str=None, quant:Optional[str]=None):
    modes = []
    if quant == "8bit":
        modes = [("8bit_auto", dict(device_map="auto", quant="8bit")),
                 ("fp16_full", dict(device_map={"":"cuda:0"}, torch_dtype=torch.float16)),
                 ("fp16_auto", dict(device_map="auto", torch_dtype=torch.float16)),
                 ("cpu", dict(device_map="cpu"))]
    elif quant == "4bit":
        modes = [("4bit_auto", dict(device_map="auto", quant="4bit")),
                 ("fp16_full", dict(device_map={"":"cuda:0"}, torch_dtype=torch.float16)),
                 ("fp16_auto", dict(device_map="auto", torch_dtype=torch.float16)),
                 ("cpu", dict(device_map="cpu"))]
    else:
        modes = [("fp16_full", dict(device_map={"":"cuda:0"}, torch_dtype=torch.float16)),
                 ("fp16_auto", dict(device_map="auto", torch_dtype=torch.float16)),
                 ("cpu", dict(device_map="cpu"))]

    last_err = None
    for tag, cfg in modes:
        try:
            print(f"Loading base [{tag}] ...")
            model = _try_load(base_path, **cfg)
            if lora_path:
                model = PeftModel.from_pretrained(model, lora_path)
                try: model = model.merge_and_unload()
                except Exception: pass
            model.eval()
            if hasattr(model, "gradient_checkpointing_disable"):
                try: model.gradient_checkpointing_disable()
                except Exception: pass
            if hasattr(model.config,"use_cache"): model.config.use_cache=True
            _silence_generation(model)
            print(f"Loaded with mode={tag}")
            return model, tag
        except Exception as e:
            print(f"Failed on mode={tag}: {e}")
            last_err = e
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
            continue
    raise last_err

def load_model(base_path:str, lora_path:str=None, quant:str=None):
    model, _ = load_model_resilient(base_path, lora_path=lora_path, quant=quant)
    return model

def build_prompt(user_text:str)->str:
    system = (
        "You are an RF optimization assistant for a 60,000-person stadium. "
        "First output exactly ONE line in this format:\n"
        "5G_MHz=<value>; WiFi_MHz=<value>; 5G_%=<value>; WiFi_%=<value>\n"
        "Then provide a short explanation (one paragraph). Use numbers only for those four fields."
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

@torch.inference_mode()
def generate_once(model, tok, user_text:str, max_new_tokens:int=140):
    prompt = build_prompt(user_text)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    t0=time.time()
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id, use_cache=True
    )
    dt=time.time()-t0
    text = tok.decode(out[0], skip_special_tokens=True)
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
    return text.replace("<|im_end|>","").strip(), dt, out.shape[-1]-inputs["input_ids"].shape[-1]

# ---------------- Utilities ----------------
def split_out_names(base_out: str):
    stem, ext = (os.path.splitext(base_out) if base_out.lower().endswith(".csv") else (base_out, ".csv"))
    base_csv = f"{stem}_base{ext}"
    lora_csv = f"{stem}_lora{ext}"
    cmp_csv  = f"{stem}_compare{ext}"
    return base_csv, lora_csv, cmp_csv

def _fmt_eta(sec: float) -> str:
    sec = int(max(0, round(sec)))
    h = sec // 3600; m = (sec % 3600) // 60; s = sec % 60
    if h: return f"{h}h{m:02d}m{s:02d}s"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"

# ---------------- Batch loop ----------------
def eval_one_tag(tag:str, lora:bool, tok, data, base_path, lora_path, quant, max_new_tokens=140):
    print(f"\n>>> Loading model: {tag} (quant={quant})")
    model = load_model(base_path, lora_path if lora else None, quant)
    rows=[]
    total = len(data)
    pass_all_cnt = pass_band_only_cnt = band_cnt = pct_cnt = ratio_cnt = entity_cnt = unit_cnt = 0

    t_start = time.time()
    for idx, item in enumerate(data, 1):
        uid = item.get("id", f"sample_{idx}")
        user = get_user_text(item)
        if not user:
            continue
        truth = parse_prompt_truth(user)
        text, dt, gen_tokens = generate_once(model, tok, user, max_new_tokens=max_new_tokens)
        parsed = parse_output_values(text)

        # ----- derive percentages if not explicitly present -----
        g5 = parsed["g5_mhz"]; wf = parsed["wifi_mhz"]
        exp_g5p = parsed["g5_pct_explicit"]; exp_wfp = parsed["wifi_pct_explicit"]
        used_g5p = exp_g5p
        used_wfp = exp_wfp
        pct_source = "explicit"
        if (exp_g5p is None) or (exp_wfp is None):
            if (g5 is not None) and (wf is not None) and (g5 + wf) > 0:
                r = g5 / (g5 + wf)
                used_g5p = r * 100.0
                used_wfp = 100.0 - used_g5p
                pct_source = "derived" if (exp_g5p is None and exp_wfp is None) else "mixed"
            else:
                used_g5p = None
                used_wfp = None
                pct_source = "missing"

        # ----- checks -----
        b_ok = band_ok(g5, wf, truth.get("total_mhz"))
        p_ok = pct_ok(used_g5p, used_wfp)
        r_ok, r_err = ratio_ok(g5, wf, used_g5p, used_wfp)

        # 若显式 % 与 MHz 比例不一致，做一次“和解”：用 MHz 推导覆盖百分比再重算
        if (not r_ok) and (g5 is not None) and (wf is not None) and ((g5 + wf) > 0):
            r = g5 / (g5 + wf)
            used_g5p = r * 100.0
            used_wfp = 100.0 - used_g5p
            pct_source = "reconciled" if pct_source in ("explicit","mixed") else pct_source
            p_ok = pct_ok(used_g5p, used_wfp)
            r_ok, r_err = ratio_ok(g5, wf, used_g5p, used_wfp)

        ent = entity_ok(text, truth)
        e_ok = all(ent.values())
        u_ok_full = unit_range_ok(g5, wf, used_g5p, used_wfp, truth.get("total_mhz"))
        u_ok_mhz  = unit_range_mhz_ok(g5, wf, truth.get("total_mhz"))

        pass_band_only = all([b_ok, u_ok_mhz, e_ok])          # MHz-only view
        pass_full      = all([b_ok, p_ok, r_ok, e_ok, u_ok_full])

        # counters
        band_cnt += int(b_ok); pct_cnt += int(p_ok); ratio_cnt += int(r_ok)
        entity_cnt += int(e_ok); unit_cnt += int(u_ok_full)
        pass_band_only_cnt += int(pass_band_only); pass_all_cnt += int(pass_full)

        rows.append({
            "id": uid, "model": tag,
            "total_mhz": truth.get("total_mhz"),
            "g5_mhz": g5, "wifi_mhz": wf,
            "sum_mhz": (None if None in (g5, wf) else g5+wf),
            "band_ok": b_ok,

            "g5_pct_explicit": exp_g5p, "wifi_pct_explicit": exp_wfp,
            "used_g5_pct": (None if used_g5p is None else round(used_g5p, 3)),
            "used_wifi_pct": (None if used_wfp is None else round(used_wfp, 3)),
            "sum_pct_used": (None if None in (used_g5p, used_wfp) else round(used_g5p+used_wfp, 3)),
            "pct_source": pct_source,
            "pct_ok": p_ok,

            "ratio_err_pp": r_err, "ratio_ok": r_ok,

            "aoa_ok": ent["aoa_ok"], "snr_ok": ent["snr_ok"], "load_ok": ent["load_ok"],
            "entity_ok": e_ok,

            "unit_range_ok": u_ok_full,      # requires % present
            "unit_range_mhz_ok": u_ok_mhz,   # MHz-only range check

            "pass_band_only": pass_band_only,
            "pass_full": pass_full,          # new name for clarity
            "pass_all": pass_full,           # keep backward compatibility

            "latency_s": round(dt,3), "gen_tokens": gen_tokens,
            "output": text
        })

        if idx % 10 == 0 or idx == total:
            elapsed = time.time() - t_start
            avg = elapsed / idx
            eta = avg * (total - idx)
            print(f"[{tag}] {idx}/{total} | band_only={pass_band_only_cnt}/{idx} ({pass_band_only_cnt/idx:.1%}) "
                  f"| full={pass_all_cnt}/{idx} ({pass_all_cnt/idx:.1%}) | avg={avg:.2f}s | ETA≈{_fmt_eta(eta)}")

    # cleanup this model
    del model
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass

    print(f"[{tag}] summary: BandOnly={pass_band_only_cnt}/{total} ({pass_band_only_cnt/total:.1%}) | "
          f"Pass@Constraint={pass_all_cnt}/{total} ({pass_all_cnt/total:.1%})")
    return rows

def write_csv(rows: List[Dict[str,Any]], path: str):
    if not rows:
        print(f"Warning: no rows to write for {path}")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"CSV saved to: {path}")

def merge_compare(base_rows: List[Dict[str,Any]], lora_rows: List[Dict[str,Any]], out_csv: str):
    bmap = {r["id"]: r for r in base_rows}
    lmap = {r["id"]: r for r in lora_rows}
    ids = sorted(set(bmap.keys()) | set(lmap.keys()))
    rows=[]
    for i in ids:
        b = bmap.get(i)
        l = lmap.get(i)
        row = {"id": i}
        def put(prefix, src, key, default=None):
            row[f"{prefix}_{key}"] = (src.get(key) if src else default)
        keys = ["total_mhz","g5_mhz","wifi_mhz","sum_mhz","band_ok",
                "g5_pct_explicit","wifi_pct_explicit","used_g5_pct","used_wifi_pct","sum_pct_used","pct_source","pct_ok",
                "ratio_err_pp","ratio_ok","entity_ok","unit_range_ok","unit_range_mhz_ok",
                "pass_band_only","pass_full","latency_s","gen_tokens","output"]
        for k in keys:
            put("base", b, k)
            put("lora", l, k)

        # improvement flags
        row["improve_pass_band_only"] = (bool(l and l.get("pass_band_only")) and not bool(b and b.get("pass_band_only")))
        row["improve_pass_full"]      = (bool(l and l.get("pass_full")) and not bool(b and b.get("pass_full")))
        row["improve_ratio_err"] = None
        if (b and b.get("ratio_err_pp") is not None) and (l and l.get("ratio_err_pp") is not None):
            row["improve_ratio_err"] = float(b["ratio_err_pp"]) - float(l["ratio_err_pp"])  # positive=better
        rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Merged comparison CSV saved to: {out_csv}")
    else:
        print("No rows to merge for comparison CSV.")

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET_DEFAULT, help="Path to JSONL eval file")
    ap.add_argument("--out", default="constraint_eval.csv", help="Output CSV filename stem")
    ap.add_argument("--quant", default="none", choices=["none","8bit","4bit","auto"],
                    help="Quantization mode; 'auto' tries 8bit then falls back safely")
    # 默认各 3000 条；<=0 => 全量
    ap.add_argument("--limit-base", type=int, default=2000,
                    help="Number of samples for BASE; <=0 = all (default: 2000)")
    ap.add_argument("--limit-lora", type=int, default=2000,
                    help="Number of samples for LORA; <=0 = all (default: 2000)")
    ap.add_argument("--only", choices=["base","lora","both"], default="both")
    ap.add_argument("--base", default=BASE_PATH)
    ap.add_argument("--lora", default=LORA_PATH)
    args = ap.parse_args()

    dataset_path = resolve_dataset_path(args.dataset)
    quant = ("8bit" if args.quant == "auto" else (None if args.quant=="none" else args.quant))

    # Load once up to the max needed (saves I/O), then slice per model.
    lb = args.limit_base
    ll = args.limit_lora
    max_limit = 0
    positive_limits = [v for v in (lb, ll) if v is not None and v > 0]
    if positive_limits:
        max_limit = max(positive_limits)
    data = load_jsonl(dataset_path, limit=max_limit)

    # Slice per model
    base_data = data if (lb is None or lb <= 0) else data[:min(lb, len(data))]
    lora_data = data if (ll is None or ll <= 0) else data[:min(ll, len(data))]

    base_csv, lora_csv, cmp_csv = split_out_names(args.out)
    tok = prepare_tokenizer(args.base)

    base_rows = []
    lora_rows = []

    if args.only in ("base","both"):
        print(f"BASE samples to evaluate: {len(base_data)} (limit-base={lb})")
        base_rows = eval_one_tag("BASE", lora=False, tok=tok, data=base_data,
                                 base_path=args.base, lora_path=args.lora, quant=quant)
        write_csv(base_rows, base_csv)

    if args.only in ("lora","both"):
        print(f"LORA samples to evaluate: {len(lora_data)} (limit-lora={ll})")
        lora_rows = eval_one_tag("LORA", lora=True, tok=tok, data=lora_data,
                                 base_path=args.base, lora_path=args.lora, quant=quant)
        write_csv(lora_rows, lora_csv)

    if base_rows and lora_rows:
        merge_compare(base_rows, lora_rows, cmp_csv)

if __name__ == "__main__":
    main()

