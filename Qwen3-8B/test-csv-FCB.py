# -*- coding: utf-8 -*-
"""
FCB evaluation for sector-wise MHz allocation (Base vs LoRA) — fp16_full preferred

Dataset format (JSONL):
  Each line: {"id": ..., "prompt": "<text>", ...}
  - We read "prompt" as user text; if missing, fallback to common fields.

Model output format (enforced by system prompt):
  A single JSON object in the FIRST line only, like:
  {
    "A": {"NR-U": 100.0, "WiFi6": 0.0},
    "B": {"NR-U": 100.0, "WiFi6": 0.0},
    "C": {"NR-U": 95.0,  "WiFi6": 5.0},
    "D": {"NR-U": 100.0, "WiFi6": 0.0}
  }
  Numbers are MHz; for each sector, NR-U + WiFi6 = SECTOR_TOTAL (default 100).

We check:
  - json_ok: first JSON object parsed successfully
  - keys_ok: sectors present and contain "NR-U" & "WiFi6" (with synonym normalization)
  - per-sector: sum_ok (≈ SECTOR_TOTAL within tolerance), range_ok (0..SECTOR_TOTAL)
  - pass_all: json_ok & keys_ok & all sector checks

Now defaults:
  --limit-base 0 and --limit-lora 0  => run ALL items by default.
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

# ---------------- Paths ----------------
BASE_PATH = "/root/autodl-tmp/Qwen/Qwen3-8B"   # 基座
LORA_PATH        = "./qwen3-8B-ft-v2"                 # 你微调好的LoRA输出目录
DATASET_DEFAULT       = "FCB_dataset.jsonl"

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------- Helpers ----------------
NUM = r"\d+(?:,\d{3})*(?:\.\d+)?"

def to_float_any(x)->Optional[float]:
    """Convert numbers like '95', '95.0', '95%', '95,000' to float."""
    if x is None: return None
    if isinstance(x, (int, float)): return float(x)
    if not isinstance(x, str): return None
    s = x.strip()
    if s.endswith("%"): s = s[:-1]
    s = s.replace(",", "")
    try:
        return float(s)
    except:
        return None

def resolve_dataset_path(path: str) -> str:
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
    path = resolve_dataset_path(path)
    data=[]
    with open(path,"r",encoding="utf-8") as f:
        for _, line in enumerate(f,1):
            line=line.strip()
            if not line: continue
            try:
                obj=json.loads(line)
                data.append(obj)
                if (limit is not None) and (limit > 0) and len(data)>=limit:
                    break
            except:
                pass
    return data

# -------- robust user text extraction (prefer FCB's 'prompt') --------
_SINGLE_PROMPT_KEYS = ("prompt","instruction","input","question","query","text","user")
def get_user_text(item:Dict[str,Any])->Optional[str]:
    for k in _SINGLE_PROMPT_KEYS:
        v = item.get(k)
        if isinstance(v,str) and v.strip():
            return v.strip()
    # fall back to conversations-like layouts
    for list_key in ("conversations","messages","dialogue","turns"):
        conv = item.get(list_key)
        if isinstance(conv, list) and conv:
            t0 = conv[0]
            if isinstance(t0, dict):
                for tk in ("value","content","text","utterance"):
                    v = t0.get(tk)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
    return None

def filter_items_with_user(data: List[Dict[str,Any]], peek:int=10)->List[Dict[str,Any]]:
    out=[]; miss=[]
    for it in data:
        if get_user_text(it): out.append(it)
        else: miss.append(it.get("id"))
    if miss:
        print(f"[INFO] Filtered items without user text: {len(miss)} (sample ids: {miss[:peek]})")
    print(f"[INFO] Using {len(out)}/{len(data)} items after filtering.")
    return out

# ---------------- JSON extraction ----------------
def extract_json_head(text: str, max_chars: int = 1500) -> Tuple[bool, Optional[dict], Optional[str]]:
    """
    Extract the FIRST JSON object from model output (first line preferred),
    tolerant to code fences and extra prose after.
    Returns (ok, obj, raw_fragment)
    """
    s = text.lstrip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1: s = s[nl+1:]
    s = s[:max_chars]
    st = s.find("{")
    if st < 0: return False, None, None
    ss = s[st:]
    depth=0; in_str=False; esc=False
    for i,ch in enumerate(ss):
        if in_str:
            if esc: esc=False
            elif ch=="\\": esc=True
            elif ch=="\"": in_str=False
        else:
            if ch=="\"": in_str=True
            elif ch=="{": depth+=1
            elif ch=="}":
                depth-=1
                if depth==0:
                    frag = ss[:i+1]
                    try:
                        obj=json.loads(frag)
                        return True, obj, frag
                    except Exception:
                        return False, None, frag
    return False, None, None

# ---------------- Allocation normalization ----------------
_SERVICE_SYNS = {
    "nr-u":"NR-U", "nru":"NR-U", "nr u":"NR-U", "5g":"NR-U", "5gnr":"NR-U", "5g nr-u":"NR-U",
    "wifi6":"WiFi6", "wi-fi6":"WiFi6", "wi-fi 6":"WiFi6", "wifi 6":"WiFi6", "wi fi 6":"WiFi6"
}
def norm_service_key(k:str)->str:
    ks = re.sub(r"[\s_\-]+"," ", k.strip().lower())
    ks = ks.replace("nr u","nr-u")
    return _SERVICE_SYNS.get(ks, k.strip())

def normalize_alloc(obj:dict) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Expect obj like:
      {"A":{"NR-U":100,"WiFi6":0}, "B":{...}, "C":{...}, "D":{...}}
    Returns dict with normalized service keys & float values.
    """
    if not isinstance(obj, dict): return None
    out={}
    for sec, sub in obj.items():
        if not isinstance(sec, str): continue
        if not isinstance(sub, dict): continue
        sec_key = sec.strip().upper()
        sub_out={}
        for k,v in sub.items():
            if not isinstance(k, str): continue
            k2 = norm_service_key(k)
            val = to_float_any(v)
            if val is not None:
                sub_out[k2]=val
        if sub_out:
            out[sec_key]=sub_out
    return out if out else None

# ---------------- Checks ----------------
def sector_checks(alloc: Dict[str, Dict[str,float]],
                  sectors: List[str], sector_total: float, tol: float)->Tuple[Dict[str,bool], bool]:
    """
    For each sector: need both keys 'NR-U' and 'WiFi6', sums to sector_total±tol, and in [0, sector_total]
    """
    results: Dict[str,bool] = {}
    keys_ok=True
    all_ok=True
    for s in sectors:
        d = alloc.get(s, {})
        has = ("NR-U" in d) and ("WiFi6" in d)
        results[f"{s}_has_keys"] = has
        keys_ok = keys_ok and has

        if has:
            nr = d["NR-U"]; wf = d["WiFi6"]
            sum_ok = (abs((nr + wf) - sector_total) <= tol)
            rng_ok = (0.0 <= nr <= sector_total) and (0.0 <= wf <= sector_total)
            results[f"{s}_sum_ok"] = sum_ok
            results[f"{s}_range_ok"] = rng_ok
            all_ok = all_ok and sum_ok and rng_ok
        else:
            results[f"{s}_sum_ok"] = False
            results[f"{s}_range_ok"] = False
            all_ok = False
    results["keys_ok"] = keys_ok
    return results, all_ok and keys_ok

# ---------------- Model loading & generation ----------------
def prepare_tokenizer(base_path:str):
    tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True, use_fast=False, padding_side="right")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return tok

def _mode_config(tag:str):
    if tag == "fp16_full":
        return dict(device_map={"":"cuda:0"}, torch_dtype=torch.float16)
    if tag == "fp16_auto":
        return dict(device_map="auto", torch_dtype=torch.float16)
    if tag == "8bit_auto":
        return dict(device_map="auto", quant="8bit")
    if tag == "4bit_auto":
        return dict(device_map="auto", quant="4bit")
    if tag == "cpu":
        return dict(device_map="cpu")
    raise ValueError(f"Unknown mode tag: {tag}")

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

def load_model_resilient(base_path:str, lora_path:str=None, mode_sequence:List[str]=None, strict:bool=False):
    """
    Try modes in order given by mode_sequence. If strict=True, only try the first one.
    """
    if not mode_sequence:
        mode_sequence = ["fp16_full","fp16_auto","8bit_auto","4bit_auto","cpu"]
    if strict:
        mode_sequence = [mode_sequence[0]]

    last_err = None
    for tag in mode_sequence:
        cfg = _mode_config(tag)
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

def load_model(base_path:str, lora_path:str=None, mode_sequence:List[str]=None, strict:bool=False):
    model, tag = load_model_resilient(base_path, lora_path=lora_path, mode_sequence=mode_sequence, strict=strict)
    return model

# ---------------- Prompt ----------------
def build_system_hint(sectors: List[str], sector_total: float)->str:
    secs = ", ".join(sectors)
    return (
        "You are an RF allocation assistant. "
        f"Output EXACTLY ONE JSON object on the FIRST line only. Sectors: [{secs}]. "
        f"For EACH sector, provide two keys: \"NR-U\" and \"WiFi6\" (numbers = MHz). "
        f"For EACH sector: NR-U + WiFi6 MUST equal {sector_total} (MHz). "
        "Numbers only (no units or %). After the JSON, add one short sentence of rationale."
    )

def build_prompt(user_text:str, system_text:str)->str:
    return (
        f"<|im_start|>system\n{system_text}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

@torch.inference_mode()
def generate_once(model, tok, user_text:str, system_text:str, max_new_tokens:int=120):
    prompt = build_prompt(user_text, system_text)
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

# ---------------- CSV helpers ----------------
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
def eval_one_tag(tag:str, lora:bool, tok, data, base_path, lora_path,
                 mode_sequence:List[str], strict_mode:bool,
                 sectors: List[str], sector_total: float, tol_mhz: float,
                 max_new_tokens:int=120):
    print(f"\n>>> Loading model: {tag}")
    model = load_model(base_path, lora_path if lora else None,
                       mode_sequence=mode_sequence, strict=strict_mode)
    rows=[]
    total = len(data)
    pass_all_cnt = json_ok_cnt = 0

    system_text = build_system_hint(sectors, sector_total)

    t_start = time.time()
    for idx, item in enumerate(data, 1):
        uid = item.get("id", f"sample_{idx}")
        user = get_user_text(item)
        if not user:
            continue

        text, dt, gen_tokens = generate_once(model, tok, user_text=user,
                                             system_text=system_text,
                                             max_new_tokens=max_new_tokens)

        json_ok, obj, frag = extract_json_head(text, max_chars=1500)
        alloc = normalize_alloc(obj) if json_ok else None

        # checks
        keys_ok = False
        per_sec_results = {}
        per_sec_all_ok = False
        if alloc is not None:
            per_sec_results, per_sec_all_ok = sector_checks(alloc, sectors, sector_total, tol_mhz)
            keys_ok = per_sec_results.get("keys_ok", False)

        pass_all = bool(json_ok and keys_ok and per_sec_all_ok)
        pass_all_cnt += int(pass_all)
        json_ok_cnt  += int(json_ok)

        # flatten per-sector values for CSV
        flat_vals={}
        for s in sectors:
            nr = alloc.get(s,{}).get("NR-U") if alloc else None
            wf = alloc.get(s,{}).get("WiFi6") if alloc else None
            flat_vals[f"{s}_NRU_mhz"] = nr
            flat_vals[f"{s}_WiFi6_mhz"] = wf
            if (nr is not None) and (wf is not None):
                flat_vals[f"{s}_sum_mhz"] = nr + wf
            else:
                flat_vals[f"{s}_sum_mhz"] = None

        row = {
            "id": uid, "model": tag,
            "json_ok": json_ok, "keys_ok": keys_ok, "pass_all": pass_all,
            "latency_s": round(dt,3), "gen_tokens": gen_tokens,
            "output": text
        }
        row.update(flat_vals)
        if per_sec_results:
            row.update(per_sec_results)
        rows.append(row)

        if idx % 10 == 0 or idx == total:
            elapsed = time.time() - t_start
            avg = elapsed / idx
            eta = avg * (total - idx)
            print(f"[{tag}] {idx}/{total} | json_ok={json_ok_cnt}/{idx} ({json_ok_cnt/idx:.1%}) "
                  f"| pass_all={pass_all_cnt}/{idx} ({pass_all_cnt/idx:.1%}) "
                  f"| avg={avg:.2f}s | ETA≈{_fmt_eta(eta)}")

    del model
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass

    print(f"[{tag}] summary: PassAll={pass_all_cnt}/{total} ({pass_all_cnt/total:.1%}) | "
          f"JSON_OK={json_ok_cnt}/{total} ({json_ok_cnt/total:.1%})")
    return rows

def write_csv(rows: List[Dict[str,Any]], path: str):
    if not rows:
        print(f"Warning: no rows to write for {path}")
        return
    all_keys=set()
    for r in rows: all_keys.update(r.keys())
    base_keys = ["id","model","pass_all","json_ok","keys_ok","latency_s","gen_tokens"]
    fieldnames = base_keys + sorted([k for k in all_keys if k not in base_keys and k!="output"]) + ["output"]
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
        keys = set()
        if b: keys.update(b.keys())
        if l: keys.update(l.keys())
        for k in sorted(keys):
            if k=="id": continue
            put("base", b, k)
            put("lora", l, k)
        base_ok = bool(b and b.get("pass_all"))
        lora_ok = bool(l and l.get("pass_all"))
        row["improve_pass_all"] = (lora_ok and not base_ok)
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
    ap.add_argument("--dataset", default=DATASET_DEFAULT, help="Path to FCB JSONL file")
    ap.add_argument("--out", default="fcb_eval.csv", help="Output CSV filename stem")
    ap.add_argument("--limit-base", type=int, default=0,
                    help="Number of samples for BASE; <=0 = ALL (default: 0)")
    ap.add_argument("--limit-lora", type=int, default=0,
                    help="Number of samples for LORA; <=0 = ALL (default: 0)")
    ap.add_argument("--only", choices=["base","lora","both"], default="both")
    ap.add_argument("--base", default=BASE_PATH)
    ap.add_argument("--lora", default=LORA_PATH)

    # Preferred loading mode controls (fp16_full by default)
    ap.add_argument("--mode", default="fp16_full",
                    choices=["fp16_full","fp16_auto","8bit_auto","4bit_auto","cpu","auto"],
                    help="Preferred loading mode. 'auto' tries fp16_full->fp16_auto->8bit->4bit->cpu (default: fp16_full)")
    ap.add_argument("--strict-mode", action="store_true",
                    help="If set, do NOT fallback when the chosen mode fails.")

    # FCB-specific evaluation knobs
    ap.add_argument("--sectors", default="A,B,C,D",
                    help="Comma-separated sector list to require (default: A,B,C,D)")
    ap.add_argument("--sector-total", type=float, default=100.0,
                    help="Per-sector total MHz that NR-U + WiFi6 must sum to (default: 100)")
    ap.add_argument("--tol-mhz", type=float, default=1.0,
                    help="Tolerance when checking sector sums (MHz, default: 1.0)")
    ap.add_argument("--max-new-tokens", type=int, default=120,
                    help="Generation cap; 80–120 is usually sufficient")

    args = ap.parse_args()

    dataset_path = resolve_dataset_path(args.dataset)

    sectors = [s.strip().upper() for s in args.sectors.split(",") if s.strip()]
    if not sectors:
        sectors = ["A","B","C","D"]

    # Build loading sequence
    if args.mode == "auto":
        mode_sequence = ["fp16_full","fp16_auto","8bit_auto","4bit_auto","cpu"]
    else:
        mode_sequence = [args.mode, "fp16_auto","8bit_auto","4bit_auto","cpu"] if not args.strict_mode else [args.mode]

    # Load once up to the max needed (saves I/O). If both limits <=0, load ALL.
    lb, ll = args.limit_base, args.limit_lora
    positive_limits = [v for v in (lb, ll) if v is not None and v > 0]
    max_limit = max(positive_limits) if positive_limits else 0
    raw = load_jsonl(dataset_path, limit=max_limit if max_limit>0 else None)
    data_all = filter_items_with_user(raw)

    base_data = data_all if (lb is None or lb <= 0) else data_all[:min(lb, len(data_all))]
    lora_data = data_all if (ll is None or ll <= 0) else data_all[:min(ll, len(data_all))]

    base_csv, lora_csv, cmp_csv = split_out_names(args.out)
    tok = prepare_tokenizer(args.base)

    base_rows = []
    lora_rows = []

    if args.only in ("base","both"):
        print(f"BASE samples to evaluate: {len(base_data)} (limit-base={lb})")
        base_rows = eval_one_tag("BASE", lora=False, tok=tok, data=base_data,
                                 base_path=args.base, lora_path=args.lora,
                                 mode_sequence=mode_sequence, strict_mode=args.strict_mode,
                                 sectors=sectors, sector_total=args.sector_total, tol_mhz=args.tol_mhz,
                                 max_new_tokens=args.max_new_tokens)
        write_csv(base_rows, base_csv)

    if args.only in ("lora","both"):
        print(f"LORA samples to evaluate: {len(lora_data)} (limit-lora={ll})")
        lora_rows = eval_one_tag("LORA", lora=True, tok=tok, data=lora_data,
                                 base_path=args.base, lora_path=args.lora,
                                 mode_sequence=mode_sequence, strict_mode=args.strict_mode,
                                 sectors=sectors, sector_total=args.sector_total, tol_mhz=args.tol_mhz,
                                 max_new_tokens=args.max_new_tokens)
        write_csv(lora_rows, lora_csv)

    if base_rows and lora_rows:
        merge_compare(base_rows, lora_rows, cmp_csv)

if __name__ == "__main__":
    main()
