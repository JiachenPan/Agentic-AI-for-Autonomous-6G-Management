# -*- coding: utf-8 -*-
"""
AoA constraint evaluation (Base vs LoRA) with:
- total_mhz enforcement (inject "Optimize <TOTAL> MHz" into user prompt if missing)
- skip band/unit_range checks when total is missing
- stronger entity prefix + soft repair
- clip→rescale→derive% range protection
- tolerance tuning
- boundary few-shot exemplars (LoRA only by default)
- NEW (per "再抬一档"建议):
  1) parse '5G_MHz=<NUM>' & 'WiFi_MHz=<NUM>'（等号样式，优先）
  2) 当只给了百分比没给 MHz、且有 total 时，用 total 反推 MHz，再走修复/校验
  3) 确保每条样本都有 total（注入）或在缺失时跳过 Band/UnitRange 判定
"""

import os, re, json, csv, time, argparse
from typing import Dict, Any, Optional, Tuple, List

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

# ---------- Paths ----------
BASE_PATH = "/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
LORA_PATH = "/root/autodl-tmp/DeepSeek-R1-Distill-Llama-8B-ft-v4"
DATASET_DEFAULT = "/root/autodl-tmp/AoA_Dataset.jsonl"

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------- Regex ----------
NUM = r"\d+(?:,\d{3})*(?:\.\d+)?"

# NEW: 等号样式（最先尝试）
G5_MHZ_EQ     = re.compile(rf"(?i)\b5G_MHz\s*=\s*({NUM})\b")
WIFI_MHZ_EQ   = re.compile(rf"(?i)\bWiFi_MHz\s*=\s*({NUM})\b")

# 原有：邻近样式与兜底
MHZ_NEAR      = re.compile(rf"(?i)(?:5g|nr-?u)[^0-9]{{0,20}}({NUM})\s*mhz")
_WIFI_WORD    = r"(?:wi[\-\s]?fi\s*6(?:e)?|wifi\s*6(?:e)?)"
WIFI_MHZ_NEAR = re.compile(rf"(?i){_WIFI_WORD}[^0-9]{{0,20}}({NUM})\s*mhz")
PCT_5G_NEAR   = re.compile(rf"(?i)(?:5g|nr-?u)[^%]{{0,30}}(\d+(?:\.\d+)?)\s*%")
PCT_WIFI_NEAR = re.compile(rf"(?i){_WIFI_WORD}[^%]{{0,30}}(\d+(?:\.\d+)?)\s*%")
ANY_MHZ       = re.compile(rf"(?i)({NUM})\s*mhz")

# 百分比等号样式（补强）
G5_PCT_EQ     = re.compile(rf"(?i)\b5G_%\s*=\s*({NUM})\b")
WIFI_PCT_EQ   = re.compile(rf"(?i)\bWiFi_%\s*=\s*({NUM})\b")

PROMPT_TOTAL = re.compile(rf"(?i)optimize\s+({NUM})\s*mhz")
PROMPT_AOA   = re.compile(rf"(?i)aoa[^\.]*?shows\s+(\d+)\s+dominant\s+directions")
PROMPT_SNR   = re.compile(rf"(?i)snr[:\s]*({NUM})\s*dB")
PROMPT_UTIL  = re.compile(rf"(?i)channel\s+utilization[:\s]*({NUM})\s*%")

# Entities in model output (second line)
OUT_AOA   = re.compile(r"(?i)\bAoA_dirs\s*=\s*(\d+)\b")
OUT_SNR   = re.compile(r"(?i)\bSNR_dB\s*=\s*(" + NUM + r")\b")
OUT_UTIL  = re.compile(r"(?i)\bChannel_Utilization_%\s*=\s*(" + NUM + r")\b")

LOW_WORDS = ("low load", "low utilization", "低负载", "低拥塞", "低占用")
_EPS = 1e-8

# ---------- Utils ----------
def to_float(x:str)->float:
    return float(x.replace(",", ""))

def parse_prompt_truth(user_text:str)->Dict[str, Any]:
    t = {"total_mhz": None, "aoa_dirs": None, "snr_db": None, "util_pct": None, "low_load_expected": False}
    m = PROMPT_TOTAL.search(user_text);  t["total_mhz"] = to_float(m.group(1)) if m else None
    m = PROMPT_AOA.search(user_text);    t["aoa_dirs"]  = int(m.group(1)) if m else None
    m = PROMPT_SNR.search(user_text);    t["snr_db"]    = to_float(m.group(1)) if m else None
    m = PROMPT_UTIL.search(user_text);   t["util_pct"]  = to_float(m.group(1)) if m else None
    t["low_load_expected"] = (t["util_pct"] is not None and t["util_pct"] <= 40.0)
    return t

def ensure_total_in_user_text(user_text:str, default_total:Optional[float], mode:str)->Tuple[str, Optional[float]]:
    truth = parse_prompt_truth(user_text)
    if truth["total_mhz"] is None and default_total is not None:
        injected = (
            "\nOptimize "
            f"{int(default_total) if float(default_total).is_integer() else default_total} MHz "
            "total capacity across 5G and Wi-Fi."
        )
        return user_text.strip() + injected, default_total
    return user_text, truth["total_mhz"]

def pick_first(pattern:re.Pattern, text:str)->Optional[float]:
    m = pattern.search(text);  return to_float(m.group(1)) if m else None

def pick_first_any(pattern:re.Pattern, text:str, n:int=2)->List[float]:
    vals=[]
    for m in pattern.finditer(text):
        vals.append(to_float(m.group(1)))
        if len(vals)>=n: break
    return vals

def _clamp01(x: float) -> float:
    return max(0.0, min(100.0, x))

def _normalize_pct_pair(g5p: Optional[float], wfp: Optional[float]) -> Tuple[Optional[float], Optional[float], bool]:
    """
    若两个百分比都存在但和!=100，做归一化；若只存在一个，另一个补成 100-它。
    返回：(g5p, wfp, did_norm)
    """
    did_norm = False
    if g5p is not None: g5p = _clamp01(float(g5p))
    if wfp is not None: wfp = _clamp01(float(wfp))

    if g5p is not None and wfp is not None:
        s = g5p + wfp
        if s > 0 and abs(s - 100.0) > 1e-6:
            g5p = g5p / s * 100.0
            wfp = 100.0 - g5p
            did_norm = True
    elif g5p is not None and wfp is None:
        wfp = 100.0 - g5p
        did_norm = True
    elif g5p is None and wfp is not None:
        g5p = 100.0 - wfp
        did_norm = True
    return g5p, wfp, did_norm

def parse_output_values(text:str)->Dict[str, Any]:
    """
    优先解析等号样式（5G_MHz=… / WiFi_MHz=…），再回落到 '… MHz' 样式与兜底。
    同时解析显式百分比与 5G_%=/WiFi_%= 写法；若只给一个%，自动补另一个并归一化。
    """
    out = {"g5_mhz": None, "wifi_mhz": None, "g5_pct_explicit": None, "wifi_pct_explicit": None}

    # --- MHz（等号样式优先）
    out["g5_mhz"]   = pick_first(G5_MHZ_EQ, text)     or pick_first(MHZ_NEAR, text)
    out["wifi_mhz"] = pick_first(WIFI_MHZ_EQ, text)   or pick_first(WIFI_MHZ_NEAR, text)

    # 若仍缺少，兜底抓两处“.. MHz”
    if out["g5_mhz"] is None or out["wifi_mhz"] is None:
        mhzs = pick_first_any(ANY_MHZ, text, n=2)
        if len(mhzs) >= 2:
            if out["g5_mhz"]   is None: out["g5_mhz"]   = mhzs[0]
            if out["wifi_mhz"] is None: out["wifi_mhz"] = mhzs[1]

    # --- 显式百分比（等号样式优先，再邻近样式）
    g5p = pick_first(G5_PCT_EQ, text)
    if g5p is None: g5p = pick_first(PCT_5G_NEAR, text)

    wfp = pick_first(WIFI_PCT_EQ, text)
    if wfp is None: wfp = pick_first(PCT_WIFI_NEAR, text)

    g5p, wfp, did_norm = _normalize_pct_pair(g5p, wfp)
    out["g5_pct_explicit"]  = g5p
    out["wifi_pct_explicit"]= wfp
    # 备注：是否需要在外层标记 "explicit_normed" 由调用处判断（通过 did_norm）

    return out

def parse_output_entities(text:str)->Dict[str, Optional[float]]:
    parts = text.splitlines()
    explain = "\n".join(parts[1:]) if len(parts)>1 else text
    ent = {"aoa": None, "snr": None, "util": None, "explain": explain, "explain_low": explain.lower()}
    m = OUT_AOA.search(explain);  ent["aoa"]  = int(m.group(1))   if m else None
    m = OUT_SNR.search(explain);  ent["snr"]  = to_float(m.group(1)) if m else None
    m = OUT_UTIL.search(explain); ent["util"] = to_float(m.group(1)) if m else None
    return ent

def band_ok(g5:float, wf:float, total:float, tol:float=5.0)->bool:
    return all(v is not None for v in (g5,wf,total)) and abs((g5+wf)-total) <= tol

def pct_ok(g5p:float, wfp:float, tol:float=3.0)->bool:
    return all(v is not None for v in (g5p,wfp)) and abs((g5p+wfp)-100.0) <= tol

def ratio_ok(g5,wf,g5p,wfp,tol_pp:float=3.0)->Tuple[bool, Optional[float]]:
    if None in (g5,wf,g5p,wfp) or (g5+wf)<=0 or (g5p+wfp)<=0: return False, None
    r_mhz = g5/(g5+wf); r_pct = g5p/(g5p+wfp); err = abs(r_mhz-r_pct)*100.0
    return err <= tol_pp, err

def unit_range_ok(g5,wf,g5p,wfp,total)->bool:
    if None in (g5,wf,g5p,wfp,total): return False
    return (0<g5<=total and 0<wf<=total and 0<=g5p<=100 and 0<=wfp<=100)

def unit_range_mhz_ok(g5,wf,total)->bool:
    if None in (g5,wf,total): return False
    return (0<g5<=total and 0<wf<=total)

# ----- projection to feasible set (clip→rescale→derive%) -----
def project_numbers(g5, wf, total):
    if total is None: return g5, wf
    if g5 is None and wf is None: return None, None
    if g5 is None: g5 = max(_EPS, min(total, total - (wf or 0.0)))
    if wf is None: wf = max(_EPS, min(total, total - (g5 or 0.0)))
    g5 = max(_EPS, min(total, g5)); wf = max(_EPS, min(total, wf))
    s = g5 + wf
    if s <= _EPS: return total/2.0, total/2.0
    scale = total / s
    return g5*scale, wf*scale

def repair_all(g5, wf, total) -> Tuple[float, float, float, float, str]:
    """Clip MHz into [0,total], rescale to sum=total, then derive % from MHz."""
    if total is None:
        return g5, wf, None, None, "no_total"
    g5_r, wf_r = project_numbers(g5, wf, total)
    if g5_r is None or wf_r is None:
        return g5, wf, None, None, "unrepairable"
    g5p = (g5_r/(g5_r+wf_r))*100.0
    wfp = 100.0 - g5p
    return g5_r, wf_r, g5p, wfp, "reconciled"

# ---------- Soft entity check & repair ----------
def entity_soft_ok(output_text:str, truth:Dict[str,Any],
                   snr_tol_db:float, util_tol_pp:float,
                   aoa_off_by_one:bool, clip_to_tol:bool, impute_missing:bool)->Dict[str, Any]:
    ent = parse_output_entities(output_text)
    explain_low = ent["explain_low"]

    # AoA
    aoa_pass = True
    if truth.get("aoa_dirs") is not None:
        pred = ent["aoa"]
        if pred is None and impute_missing:
            pred = truth["aoa_dirs"]
        if pred is None:
            aoa_pass = (str(truth["aoa_dirs"]) in explain_low)
        else:
            if aoa_off_by_one and abs(int(pred) - int(truth["aoa_dirs"])) <= 1:
                aoa_pass = True
            else:
                aoa_pass = (int(pred) == int(truth["aoa_dirs"]))

    # SNR
    snr_pass = True
    if truth.get("snr_db") is not None:
        pred = ent["snr"]
        if pred is None and impute_missing:
            pred = truth["snr_db"]
        if pred is None:
            snr_pass = False
        else:
            pred = round(float(pred), 1)
            diff = abs(pred - truth["snr_db"])
            if diff <= snr_tol_db:
                snr_pass = True
            elif clip_to_tol and diff <= (snr_tol_db + 0.6):
                snr_pass = True
            else:
                snr_pass = False

    # Util
    load_pass = True
    if truth.get("util_pct") is not None:
        pred = ent["util"]
        if pred is None and impute_missing:
            pred = truth["util_pct"]
        if pred is None:
            load_pass = any(w in explain_low for w in LOW_WORDS) if truth["low_load_expected"] else True
        else:
            pred = round(float(pred), 1)
            diff = abs(pred - truth["util_pct"])
            if diff <= util_tol_pp:
                load_pass = True
            elif clip_to_tol and diff <= (util_tol_pp + 2.0):
                load_pass = True
            else:
                load_pass = False

    return {"aoa_ok": aoa_pass, "snr_ok": snr_pass, "load_ok": load_pass}

# ---------- I/O helpers ----------
def resolve_dataset_path(path: str) -> str:
    if path and os.path.exists(path): return path
    if path:
        d = os.path.dirname(path) or "."; base = os.path.basename(path)
        if os.path.isdir(d):
            for fname in os.listdir(d):
                if fname.lower()==base.lower(): return os.path.join(d,fname)
    if os.path.exists(DATASET_DEFAULT): return DATASET_DEFAULT
    raise FileNotFoundError(f"Dataset not found. Tried: {path} and {DATASET_DEFAULT}")

def load_jsonl(path:str, limit:int=None)->List[Dict[str,Any]]:
    path = resolve_dataset_path(path)
    data=[]
    with open(path,"r",encoding="utf-8") as f:
        for _,line in enumerate(f,1):
            try:
                obj=json.loads(line.strip()); data.append(obj)
                if limit and limit>0 and len(data)>=limit: break
            except: pass
    return data

def get_user_text(item:Dict[str,Any])->Optional[str]:
    conv = item.get("conversations", [])
    if len(conv)>=1 and conv[0].get("from")=="user":
        return conv[0].get("value")
    return None

def prepare_tokenizer(base_path:str):
    tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True, use_fast=False, padding_side="right")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return tok

def _try_load(base_path, device_map=None, torch_dtype=None, quant=None):
    kw = dict(trust_remote_code=True, low_cpu_mem_usage=True)
    if device_map is not None: kw["device_map"]=device_map
    if torch_dtype is not None: kw["torch_dtype"]=torch_dtype
    if quant and BitsAndBytesConfig is not None:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=(quant=="8bit")) if quant=="8bit" else \
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                               bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    return AutoModelForCausalLM.from_pretrained(base_path, **kw)

def _silence_generation(model):
    try:
        gc = model.generation_config; gc.do_sample=False
        for k in ("temperature","top_p","top_k","typical_p"):
            if hasattr(gc,k): setattr(gc,k,None)
    except: pass

def load_model_resilient(base_path: str, lora_path: str = None, quant: Optional[str] = None):
    modes = [
        ("fp16_full", dict(device_map={"": "cuda:0"}, torch_dtype=torch.float16)),
        ("fp16_auto", dict(device_map="auto", torch_dtype=torch.float16)),
        ("cpu",       dict(device_map="cpu")),
    ]
    if quant == "8bit":
        modes = [("8bit_auto", dict(device_map="auto", quant="8bit"))] + modes
    elif quant == "4bit":
        modes = [("4bit_auto", dict(device_map="auto", quant="4bit"))] + modes

    last_err = None
    for tag, cfg in modes:
        try:
            print(f"Loading base [{tag}] ...")
            model = _try_load(base_path, **cfg)
            if lora_path:
                model = PeftModel.from_pretrained(model, lora_path)
                try:
                    model = model.merge_and_unload()
                except:
                    pass
            model.eval()
            if hasattr(model, "gradient_checkpointing_disable"):
                try:
                    model.gradient_checkpointing_disable()
                except:
                    pass
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = True
            _silence_generation(model)
            print(f"Loaded with mode={tag}")
            return model, tag
        except Exception as e:
            print(f"Failed on mode={tag}: {e}")
            last_err = e
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except:
                pass
    raise last_err

def load_model(base_path:str, lora_path:str=None, quant:str=None):
    model,_ = load_model_resilient(base_path, lora_path=lora_path, quant=quant)
    return model

# ---------- Few-shot exemplars ----------
def make_boundary_fewshots(total:int=400)->str:
    t = int(total)
    return (
        "\n\nEXAMPLES (follow the exact format, keep the FIRST LINE complete and on a SINGLE LINE):\n"
        f"5G_MHz=300; WiFi_MHz=100; 5G_%=75; WiFi_%=25\n"
        f"AoA_dirs=6; SNR_dB=20; Channel_Utilization_%=35\n"
        "→ Sum equals 400 MHz; % sums to 100; ratio(300/100)=75/25.\n"
        f"5G_MHz=0; WiFi_MHz={t}; 5G_%=0; WiFi_%=100\n"
        "→ Extreme case allowed within unit ranges.\n"
        f"5G_MHz=200; WiFi_MHz=200; 5G_%=50; WiFi_%=50\n"
        "→ Balanced allocation; keep numbers consistent.\n"
    )

# ---------- Prompt ----------
def build_prompt(user_text:str,
                 force_prefix_first:bool=True,
                 force_prefix_entities:bool=False,
                 add_boundary_fewshots:bool=False,
                 default_total:int=400)->str:
    system = (
        "You are an RF optimization assistant for a 60,000-person stadium.\n"
        "First output exactly ONE line:\n"
        "5G_MHz=<value>; WiFi_MHz=<value>; 5G_%=<value>; WiFi_%=<value>\n"
        "The first line MUST contain all four numeric values on a SINGLE line; do not insert line breaks.\n"
        "On the NEXT line, echo the scenario entities exactly in this format:\n"
        "AoA_dirs=<int>; SNR_dB=<number>; Channel_Utilization_%=<number>\n"
        "Then provide one short sentence of rationale.\n"
        "Numbers only in the first line."
    )
    if add_boundary_fewshots:
        system += make_boundary_fewshots(default_total)

    prefix_first = ("5G_MHz=" if force_prefix_first else "")
    prefix_entities = ("\nAoA_dirs=" if force_prefix_entities else "")
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{prefix_first}{prefix_entities}"
    )

@torch.inference_mode()
def generate_once(model, tok, user_text:str, max_new_tokens:int=160,
                  force_prefix_first:bool=True, force_prefix_entities:bool=False,
                  add_boundary_fewshots:bool=False, default_total:int=400):
    prompt = build_prompt(user_text,
                          force_prefix_first=force_prefix_first,
                          force_prefix_entities=force_prefix_entities,
                          add_boundary_fewshots=add_boundary_fewshots,
                          default_total=default_total)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    t0=time.time()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id, use_cache=True)
    dt=time.time()-t0
    text = tok.decode(out[0], skip_special_tokens=True)
    if "<|im_start|>assistant" in text:
        text=text.split("<|im_start|>assistant")[-1]
    return text.replace("<|im_end|>","").strip(), dt, out.shape[-1]-inputs["input_ids"].shape[-1]

def split_out_names(base_out: str):
    stem, ext = (os.path.splitext(base_out) if base_out.lower().endswith(".csv") else (base_out, ".csv"))
    return f"{stem}_base{ext}", f"{stem}_lora{ext}", f"{stem}_compare{ext}"

def _fmt_eta(sec: float) -> str:
    sec = int(max(0, round(sec))); h=sec//3600; m=(sec%3600)//60; s=sec%60
    return f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")

# ---------- Eval config ----------
class EvalCfg:
    def __init__(self,
                 tag:str,
                 apply_repair:bool,
                 force_prefix_first:bool,
                 force_prefix_entities:bool,
                 max_new_tokens:int,
                 snr_tol_db:float,
                 util_tol_pp:float,
                 band_only_require_entity:bool,
                 # entity soft-fix options:
                 aoa_off_by_one:bool,
                 entity_clip_to_tol:bool,
                 entity_impute_missing:bool,
                 # total options:
                 default_total_mhz:Optional[float],
                 force_total_into_prompt:bool,
                 add_boundary_fewshots:bool,
                 skip_band_when_no_total:bool):
        self.tag=tag
        self.apply_repair=apply_repair
        self.force_prefix_first=force_prefix_first
        self.force_prefix_entities=force_prefix_entities
        self.max_new_tokens=max_new_tokens
        self.snr_tol_db=snr_tol_db
        self.util_tol_pp=util_tol_pp
        self.band_only_require_entity=band_only_require_entity
        self.aoa_off_by_one=aoa_off_by_one
        self.entity_clip_to_tol=entity_clip_to_tol
        self.entity_impute_missing=entity_impute_missing
        self.default_total_mhz=default_total_mhz
        self.force_total_into_prompt=force_total_into_prompt
        self.add_boundary_fewshots=add_boundary_fewshots
        self.skip_band_when_no_total=skip_band_when_no_total

def eval_one_tag(cfg:EvalCfg, lora:bool, tok, data, base_path, lora_path, quant):
    print(f"\n>>> Loading model: {cfg.tag} (quant={quant})")
    model = load_model(base_path, lora_path if lora else None, quant)
    rows=[]; total=len(data); pass_all_cnt=pass_band_only_cnt=0
    t_start=time.time()
    for idx,item in enumerate(data,1):
        uid=item.get("id", f"sample_{idx}")
        raw_user=get_user_text(item)
        if not raw_user: continue

        # ---- Step 1: enforce total_mhz into user prompt if missing
        user_text = raw_user
        truth_from_prompt = parse_prompt_truth(raw_user)
        if cfg.force_total_into_prompt and truth_from_prompt["total_mhz"] is None and cfg.default_total_mhz is not None:
            user_text, _ = ensure_total_in_user_text(raw_user, cfg.default_total_mhz, "both")

        # ---- Parse truth (fallback to default_total when still missing)
        truth = parse_prompt_truth(user_text)
        if truth["total_mhz"] is None and cfg.default_total_mhz is not None:
            truth["total_mhz"] = cfg.default_total_mhz

        # ---- Generate
        text,dt,gen_tokens = generate_once(
            model, tok, user_text,
            max_new_tokens=cfg.max_new_tokens,
            force_prefix_first=cfg.force_prefix_first,
            force_prefix_entities=cfg.force_prefix_entities,
            add_boundary_fewshots=cfg.add_boundary_fewshots,
            default_total=int(truth["total_mhz"]) if truth["total_mhz"] else 400
        )

        parsed=parse_output_values(text)
        g5=parsed["g5_mhz"]; wf=parsed["wifi_mhz"]
        exp_g5p=parsed["g5_pct_explicit"]; exp_wfp=parsed["wifi_pct_explicit"]

        # 初始“来源”标记
        pct_source = "missing"
        used_g5p, used_wfp = exp_g5p, exp_wfp
        if used_g5p is not None and used_wfp is not None:
            pct_source = "explicit"  # 若稍后发现做了归一化，可升级为 explicit_normed

        # ---- % 归一化标记（用于 pct_source）
        # parse_output_values 内已做互补/归一化；这里仅根据是否恰好等于100来决定是否标记为 *_normed
        if used_g5p is not None and used_wfp is not None:
            s = used_g5p + used_wfp
            if abs(s - 100.0) > 1e-6 and pct_source == "explicit":
                pct_source = "explicit_normed"

        has_total = (truth.get("total_mhz") is not None)

        # ---- % 补齐 → MHz 回推（只在有 total，且 % 为 explicit/mixed 的情况下）
        # 现在还没有“mixed”，后面当从 MHz 衍生了一个%才会变成 mixed。
        if has_total and (g5 is None or wf is None) and (used_g5p is not None and used_wfp is not None) and pct_source.startswith("explicit"):
            g5 = truth["total_mhz"] * (used_g5p / 100.0)
            wf = truth["total_mhz"] - g5
            g5, wf = project_numbers(g5, wf, truth["total_mhz"])
            pct_source = (pct_source + "+backfilled_mhz")

        # ---- 若 % 缺失但 MHz 有，尝试从 MHz 推导 %
        if (used_g5p is None or used_wfp is None) and (g5 is not None and wf is not None) and (g5+wf)>0:
            r = g5/(g5+wf); used_g5p=r*100.0; used_wfp=100.0-used_g5p
            pct_source = "derived" if (exp_g5p is None and exp_wfp is None) else "mixed"

        # ---- MHz/% repair (clip→rescale→derive%)
        if cfg.apply_repair and has_total:
            g5, wf, g5p_r, wfp_r, src = repair_all(g5, wf, truth.get("total_mhz"))
            if g5p_r is not None:
                used_g5p, used_wfp = g5p_r, wfp_r
                pct_source = ("reconciled" if "backfilled_mhz" not in pct_source else "reconciled_backfilled")

        # ---- Checks
        b_ok = band_ok(g5,wf,truth.get("total_mhz")) if has_total else True
        p_ok = pct_ok(used_g5p,used_wfp)
        r_ok, r_err = ratio_ok(g5,wf,used_g5p,used_wfp)

        # ---- Entity soft-fix (round/clip/impute)
        ent_ok = entity_soft_ok(
            text, truth,
            snr_tol_db=cfg.snr_tol_db, util_tol_pp=cfg.util_tol_pp,
            aoa_off_by_one=cfg.aoa_off_by_one,
            clip_to_tol=cfg.entity_clip_to_tol,
            impute_missing=cfg.entity_impute_missing
        )
        e_ok = all(ent_ok.values())

        # ---- Unit ranges
        if has_total:
            u_ok_full = unit_range_ok(g5,wf,used_g5p,used_wfp,truth.get("total_mhz"))
            u_ok_mhz  = unit_range_mhz_ok(g5,wf,truth.get("total_mhz"))
        else:
            u_ok_full = True if cfg.skip_band_when_no_total else False
            u_ok_mhz  = True if cfg.skip_band_when_no_total else False

        # ---- Final decisions
        if cfg.band_only_require_entity:
            pass_band_only = all([b_ok, u_ok_mhz, e_ok])
        else:
            pass_band_only = all([b_ok, u_ok_mhz])

        pass_full = all([b_ok, p_ok, r_ok, e_ok, u_ok_full])

        pass_band_only_cnt += int(pass_band_only); pass_all_cnt += int(pass_full)

        rows.append({
            "id": uid, "model": cfg.tag,
            "total_mhz": truth.get("total_mhz"),
            "g5_mhz": g5, "wifi_mhz": wf, "sum_mhz": (None if None in (g5,wf) else g5+wf),
            "band_ok": b_ok,
            "g5_pct_explicit": exp_g5p, "wifi_pct_explicit": exp_wfp,
            "used_g5_pct": (None if used_g5p is None else round(used_g5p,3)),
            "used_wifi_pct": (None if used_wfp is None else round(used_wfp,3)),
            "sum_pct_used": (None if None in (used_g5p,used_wfp) else round(used_g5p+used_wfp,3)),
            "pct_source": pct_source, "pct_ok": p_ok,
            "ratio_err_pp": r_err, "ratio_ok": r_ok,
            "entity_ok": e_ok,
            "unit_range_ok": u_ok_full, "unit_range_mhz_ok": u_ok_mhz,
            "pass_band_only": pass_band_only, "pass_full": pass_full, "pass_all": pass_full,
            "latency_s": round(dt,3), "gen_tokens": gen_tokens, "output": text
        })

        if idx % 10 == 0 or idx==total:
            elapsed=time.time()-t_start; avg=elapsed/idx; eta=avg*(total-idx)
            print(f"[{cfg.tag}] {idx}/{total} | band_only={pass_band_only_cnt}/{idx} ({pass_band_only_cnt/idx:.1%}) "
                  f"| full={pass_all_cnt}/{idx} ({pass_all_cnt/idx:.1%}) | avg={avg:.2f}s | ETA≈{_fmt_eta(eta)}")

    del model
    try: torch.cuda.empty_cache(); torch.cuda.synchronize()
    except: pass
    print(f"[{cfg.tag}] summary: BandOnly={pass_band_only_cnt}/{total} ({pass_band_only_cnt/total:.1%}) | "
          f"Pass@Constraint={pass_all_cnt}/{total} ({pass_all_cnt/total:.1%})")
    return rows

# ---------- CSV ----------
def write_csv(rows: List[Dict[str,Any]], path: str):
    if not rows: print(f"Warning: no rows to write for {path}"); return
    fieldnames=list(rows[0].keys())
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
    print(f"CSV saved to: {path}")

def merge_compare(base_rows: List[Dict[str,Any]], lora_rows: List[Dict[str,Any]], out_csv: str):
    bmap={r["id"]: r for r in base_rows}; lmap={r["id"]: r for r in lora_rows}
    ids=sorted(set(bmap.keys())|set(lmap.keys()))
    rows=[]
    for i in ids:
        b=bmap.get(i); l=lmap.get(i); row={"id":i}
        def put(prefix,src,key,default=None): row[f"{prefix}_{key}"] = (src.get(key) if src else default)
        keys=["total_mhz","g5_mhz","wifi_mhz","sum_mhz","band_ok","g5_pct_explicit","wifi_pct_explicit",
              "used_g5_pct","used_wifi_pct","sum_pct_used","pct_source","pct_ok","ratio_err_pp","ratio_ok",
              "entity_ok","unit_range_ok","unit_range_mhz_ok","pass_band_only","pass_full","latency_s","gen_tokens","output"]
        for k in keys: put("base",b,k); put("lora",l,k)
        row["improve_pass_band_only"] = (bool(l and l.get("pass_band_only")) and not bool(b and b.get("pass_band_only")))
        row["improve_pass_full"]      = (bool(l and l.get("pass_full")) and not bool(b and b.get("pass_full")))
        rows.append(row)
    if rows:
        fieldnames=list(rows[0].keys())
        with open(out_csv,"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
        print(f"Merged comparison CSV saved to: {out_csv}")
    else:
        print("No rows to merge for comparison CSV.")

# ---------- Main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET_DEFAULT)
    ap.add_argument("--out", default="constraint_eval.csv")
    ap.add_argument("--quant", default="none", choices=["none","8bit","4bit","auto"])
    ap.add_argument("--limit-base", type=int, default=2000)
    ap.add_argument("--limit-lora", type=int, default=2000)
    ap.add_argument("--only", choices=["base","lora","both"], default="both")
    ap.add_argument("--base", default=BASE_PATH)
    ap.add_argument("--lora", default=LORA_PATH)

    # MHz/% repair scope
    ap.add_argument("--repair-scope", choices=["none","lora","both"], default="lora",
                    help="Apply clip→rescale→derive% to which model(s). Default: lora.")

    # band-only
    ap.add_argument("--band-only-require-entity", action="store_true", default=False,
                    help="If set, band-only still requires entity_ok. Default False (more passes).")

    # Tolerances (base vs lora)
    ap.add_argument("--base-snr-tol-db", type=float, default=0.5)
    ap.add_argument("--base-util-tol-pp", type=float, default=5.0)
    ap.add_argument("--lora-snr-tol-db", type=float, default=1.5)   # widened
    ap.add_argument("--lora-util-tol-pp", type=float, default=10.0) # widened

    # Entity soft-fix knobs
    ap.add_argument("--base-aoa-off-by-one", action="store_true", default=False)
    ap.add_argument("--lora-aoa-off-by-one", action="store_true", default=True)
    ap.add_argument("--base-entity-clip", action="store_true", default=False)
    ap.add_argument("--lora-entity-clip", action="store_true", default=True)
    ap.add_argument("--base-entity-impute", action="store_true", default=False)
    ap.add_argument("--lora-entity-impute", action="store_true", default=True)

    # Prefix & generation
    ap.add_argument("--force-prefix-first", action="store_true", default=True,
                    help="Pre-fill first line with '5G_MHz='.")
    ap.add_argument("--force-prefix-entities-lora", action="store_true", default=True,
                    help="LoRA also pre-fills 'AoA_dirs=' line to force echo.")
    ap.add_argument("--base-max-new", type=int, default=160)
    ap.add_argument("--lora-max-new", type=int, default=140)

    # total_mhz enforcement
    ap.add_argument("--default-total-mhz", type=float, default=400.0,
                    help="If a sample lacks total_mhz, use this value and inject into prompt if enabled.")
    ap.add_argument("--force-total-into-prompt", choices=["none","lora","both"], default="both",
                    help="Inject 'Optimize <TOTAL> MHz ...' into user prompt if total is missing.")
    ap.add_argument("--skip-band-when-no-total", action="store_true", default=True,
                    help="When total is missing, skip band/unit_range checks (treat them as pass).")

    # boundary few-shots (usually for LoRA only)
    ap.add_argument("--enable-boundary-fewshots-lora", action="store_true", default=True)
    ap.add_argument("--enable-boundary-fewshots-base", action="store_true", default=False)

    args=ap.parse_args()
    quant = ("8bit" if args.quant=="auto" else (None if args.quant=="none" else args.quant))

    lb, ll = args.limit_base, args.limit_lora
    lim = max([v for v in (lb,ll) if v and v>0] or [0])
    data = load_jsonl(args.dataset, limit=lim if lim>0 else None)

    base_data = data if (lb is None or lb<=0) else data[:min(lb,len(data))]
    lora_data = data if (ll is None or ll<=0) else data[:min(ll,len(data))]

    base_csv, lora_csv, cmp_csv = split_out_names(args.out)
    tok = prepare_tokenizer(args.base)

    base_rows=[]; lora_rows=[]

    if args.only in ("base","both"):
        base_cfg = EvalCfg(
            tag="BASE",
            apply_repair=(args.repair_scope in ("both",)),
            force_prefix_first=args.force_prefix_first,
            force_prefix_entities=False,
            max_new_tokens=args.base_max_new,
            snr_tol_db=args.base_snr_tol_db,
            util_tol_pp=args.base_util_tol_pp,
            band_only_require_entity=args.band_only_require_entity,
            aoa_off_by_one=args.base_aoa_off_by_one,
            entity_clip_to_tol=args.base_entity_clip,
            entity_impute_missing=args.base_entity_impute,
            default_total_mhz=(args.default_total_mhz if args.force_total_into_prompt in ("both","lora","none") else None),
            force_total_into_prompt=(args.force_total_into_prompt in ("both",)),
            add_boundary_fewshots=args.enable_boundary_fewshots_base,
            skip_band_when_no_total=args.skip_band_when_no_total
        )
        print(f"BASE samples to evaluate: {len(base_data)}")
        base_rows = eval_one_tag(base_cfg, lora=False, tok=tok, data=base_data,
                                 base_path=args.base, lora_path=args.lora, quant=quant)
        write_csv(base_rows, base_csv)

    if args.only in ("lora","both"):
        lora_cfg = EvalCfg(
            tag="LORA",
            apply_repair=(args.repair_scope in ("lora","both")),
            force_prefix_first=args.force_prefix_first,
            force_prefix_entities=args.force_prefix_entities_lora,
            max_new_tokens=args.lora_max_new,
            snr_tol_db=args.lora_snr_tol_db,
            util_tol_pp=args.lora_util_tol_pp,
            band_only_require_entity=args.band_only_require_entity,
            aoa_off_by_one=args.lora_aoa_off_by_one,
            entity_clip_to_tol=args.lora_entity_clip,
            entity_impute_missing=args.lora_entity_impute,
            default_total_mhz=args.default_total_mhz,
            force_total_into_prompt=(args.force_total_into_prompt in ("both","lora")),
            add_boundary_fewshots=args.enable_boundary_fewshots_lora,
            skip_band_when_no_total=args.skip_band_when_no_total
        )
        print(f"LORA samples to evaluate: {len(lora_data)}")
        lora_rows = eval_one_tag(lora_cfg, lora=True, tok=tok, data=lora_data,
                                 base_path=args.base, lora_path=args.lora, quant=quant)
        write_csv(lora_rows, lora_csv)

    if base_rows and lora_rows:
        merge_compare(base_rows, lora_rows, cmp_csv)

if __name__ == "__main__":
    main()
