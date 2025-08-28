# finetune_deepseek_format_lock.py
# 续训 1 epoch：把“两行硬格式 + 实体回显”学进去（优先续训已有 LoRA；无则新建小 LoRA）

import os, re, json, random
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, DataCollatorForLanguageModeling, Trainer
)
from peft import (
    LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
)

# ---------------- Paths ----------------
BASE_MODEL = "/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
V3_ADAPTER = "/root/autodl-tmp/DeepSeek-R1-Distill-Llama-8B-ft-v3"   # 续训这个；若不存在则新建
DATA_PATH  = "/root/autodl-tmp/AoA_dataset.jsonl"
OUT_DIR    = "./DeepSeek-R1-Distill-Llama-8B-ft-v4"

# ---------------- Train knobs ----------------
SEED   = 42
BATCH  = 2
ACC    = 4
EPOCHS = 1
LR     = 1e-5
MAXLEN = 1024

# ---------------- Regex for extracting entities from prompt ----------------
NUM = r"\d+(?:,\d{3})*(?:\.\d+)?"
PROMPT_TOTAL = re.compile(rf"(?i)optimize\s+({NUM})\s*mhz")
PROMPT_AOA   = re.compile(rf"(?i)aoa[^\.]*?shows\s+(\d+)\s+dominant\s+directions")
PROMPT_SNR   = re.compile(rf"(?i)snr[:\s]*({NUM})\s*dB")
PROMPT_UTIL  = re.compile(rf"(?i)channel\s+utilization[:\s]*({NUM})\s*%")

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

set_seed(SEED)

def to_float(x: str) -> float:
    return float(x.replace(",", ""))

def parse_truth(u: str):
    t = {"total": 100.0, "aoa": 6, "snr": 10.0, "util": 50.0}
    m = PROMPT_TOTAL.search(u); t["total"] = to_float(m.group(1)) if m else t["total"]
    m = PROMPT_AOA.search(u);   t["aoa"]   = int(m.group(1))       if m else t["aoa"]
    m = PROMPT_SNR.search(u);   t["snr"]   = to_float(m.group(1))  if m else t["snr"]
    m = PROMPT_UTIL.search(u);  t["util"]  = to_float(m.group(1))  if m else t["util"]
    return t

def rule_alloc(total, snr, util, aoa):
    # 一个简单的可微调规则：SNR 越高 5G 比例越大；低负载稍微多给 WiFi；AoA dirs 多也稍抬 5G
    r = min(max((snr - 5.0) / 25.0, 0.35), 0.9)
    if util <= 40: r -= 0.05
    if aoa is not None and aoa >= 8: r += 0.03
    r = min(max(r, 0.1), 0.9)
    g5  = round(total * r, 1)
    wf  = round(total - g5, 1)
    g5p = round(r * 100.0, 1)
    wfp = round(100.0 - g5p, 1)
    return g5, wf, g5p, wfp

def build_chat(user_text: str):
    t = parse_truth(user_text)
    g5, wf, g5p, wfp = rule_alloc(t["total"], t["snr"], t["util"], t["aoa"])
    sys = (
        "You are an RF optimization assistant for a 60,000-person stadium.\n"
        "First output exactly ONE line:\n"
        "5G_MHz=<value>; WiFi_MHz=<value>; 5G_%=<value>; WiFi_%=<value>\n"
        "On the NEXT line, echo the scenario entities exactly in this format:\n"
        "AoA_dirs=<int>; SNR_dB=<number>; Channel_Utilization_%=<number>\n"
        "Then provide one short sentence of rationale.\n"
        "Numbers only in the first line."
    )
    first  = f"5G_MHz={g5}; WiFi_MHz={wf}; 5G_%={g5p}; WiFi_%={wfp}"
    second = (
        f"AoA_dirs={t['aoa']}; SNR_dB={t['snr']}; "
        f"Channel_Utilization_%={t['util']}"
    )
    ans = first + "\n" + second + "\n" + "Balanced allocation with feasibility satisfied."
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": ans},
    ]

# ---------------- Dataset mapping ----------------
def get_user(item):
    conv = item.get("conversations", [])
    if conv and conv[0].get("from") == "user":
        return conv[0].get("value")
    return ""

tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

raw = load_dataset("json", data_files=DATA_PATH, split="train")

def map_to_chat(example):
    u = get_user(example)
    text = tok.apply_chat_template(build_chat(u), tokenize=False, add_generation_prompt=False)
    return {"text": text + tok.eos_token}

mapped = raw.map(map_to_chat)

def tokenize_fn(batch):
    return tok(batch["text"], truncation=True, max_length=MAXLEN, padding=False)

processed = mapped.map(tokenize_fn, batched=True, num_proc=max(os.cpu_count() or 2, 2))

# ---------------- Model (8bit) + LoRA ----------------
bnb = BitsAndBytesConfig(load_in_8bit=True)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto",
)
base = prepare_model_for_kbit_training(base)

# 续训已有适配器；找不到则新建一个小 LoRA
if V3_ADAPTER and os.path.isdir(V3_ADAPTER):
    print(f"Loading existing LoRA from: {V3_ADAPTER} (trainable)")
    model = PeftModel.from_pretrained(base, V3_ADAPTER, is_trainable=True)
else:
    print("Existing LoRA not found; creating a small new LoRA head.")
    # Llama 系列常用的线性层名
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=target_modules, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(base, lora_cfg)

# ---------------- Train ----------------
data_collator = DataCollatorForLanguageModeling(tok, mlm=False)

args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=ACC,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    save_safetensors=True,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=processed,
    data_collator=data_collator,
)

trainer.train()

# 只保存 LoRA 适配器（Peft 格式）
model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
print("Saved LoRA to:", OUT_DIR)
