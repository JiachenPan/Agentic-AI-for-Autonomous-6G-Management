# -*- coding: utf-8 -*-
"""
finetune_qwen3.py —— 结构化头部 + 严格格式提示 + 只对 assistant 计 loss（适配 ft_dataset.jsonl）
- 数据项格式：{"prompt": "...", "completion": "{...JSON by sector...}"}
- 训练时：
  1) 在 user 里加入 FORMAT INSTRUCTION，要求先输出一行汇总头，再输出原 JSON；
  2) 在 assistant 目标前自动插入一行汇总头部：5G_MHz=...; WiFi_MHz=...; 5G_%=...; WiFi_%=...
  3) 只对 assistant 段计 loss（user 段标签置 -100）
- 兼容旧版 transformers：自动过滤不被支持的 TrainingArguments 参数；在缺少 evaluation_strategy 时回退。
"""

import os, glob, re, json, random, inspect
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
)
# BitsAndBytesConfig 可能在旧版 transformers 中不可用
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

# peft 组件
from peft import LoraConfig, get_peft_model
try:
    from peft import prepare_model_for_kbit_training
except Exception:
    prepare_model_for_kbit_training = None

# ----------------- 超参 ----------------- #
MODEL_PATH = "/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DATA_PATH  = "FCB_dataset.jsonl"
OUTPUT_DIR = "./DeepSeek-R1-Distill-Llama-8B-ft-v3"
BATCH_SIZE = 1
GRAD_ACC   = 8
LR         = 1e-5
EPOCHS     = 3
LORA_RANK  = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_LEN    = 2048
EVAL_STEPS = 500
SEED       = 42

# ----------------- 回退安全的 TrainingArguments 构造 ----------------- #
def make_training_args_compatible(cfg: dict) -> TrainingArguments:
    """
    依据当前 transformers 版本的 TrainingArguments.__init__ 签名，自动过滤不被支持的字段；
    若不支持 evaluation_strategy 则：
      - 如果存在 evaluate_during_training，则设为 True，并尽量保留 eval_steps；
      - 否则关闭与评估相关的字段（load_best_model_at_end 等）。
    """
    sig = inspect.signature(TrainingArguments.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}

    d = {k: v for k, v in cfg.items() if k in accepted}

    has_eval_strategy = "evaluation_strategy" in accepted
    has_save_strategy = "save_strategy" in accepted  # 某些旧版无此项
    if has_eval_strategy and "evaluation_strategy" in cfg:
        d["evaluation_strategy"] = cfg["evaluation_strategy"]
    if has_save_strategy and "save_strategy" in cfg:
        d["save_strategy"] = cfg["save_strategy"]

    # 旧式 evaluate_during_training 回退
    if not has_eval_strategy:
        if "evaluate_during_training" in accepted:
            d["evaluate_during_training"] = True
            if "eval_steps" in accepted and "eval_steps" in cfg:
                d["eval_steps"] = cfg["eval_steps"]
        else:
            # 完全不支持训练中评估 → 关闭相关设置
            d.pop("eval_steps", None)
            if "load_best_model_at_end" in d:
                d["load_best_model_at_end"] = False

    # 可能不存在的键，统一剔除
    for maybe in ["report_to", "optim", "lr_scheduler_type", "logging_dir"]:
        if maybe not in accepted and maybe in d:
            d.pop(maybe)

    # 度量最佳模型字段保留但也要判是否支持
    if "metric_for_best_model" not in accepted and "metric_for_best_model" in d:
        d.pop("metric_for_best_model")
    if "greater_is_better" not in accepted and "greater_is_better" in d:
        d.pop("greater_is_better")

    return TrainingArguments(**d)

class PeriodicEvalCallback(TrainerCallback):
    def __init__(self, eval_steps=500):
        self.eval_steps = eval_steps
    def on_step_end(self, args, state, control, **kwargs):
        # 若当前版本不支持训练时评估，这个标记不会生效，但不会报错
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            control.should_evaluate = True
        return control

# ============ 工具：从扇区 JSON 求总 MHz + 百分比 ============ #
def parse_totals_from_json(s: str):
    """
    completion 例子：{"A":{"NR-U":1000,"WiFi6":0.0}, "B":{...}, ...}
    返回 (g5_mhz, wifi_mhz, g5_pct, wifi_pct)
    """
    try:
        obj = json.loads(s)
        g, w = 0.0, 0.0
        for _, v in obj.items():
            if isinstance(v, dict):
                g += float(v.get("NR-U", 0.0))
                w += float(v.get("WiFi6", 0.0))
        total = g + w
        g_pct = (g / total * 100.0) if total > 0 else 0.0
        w_pct = 100.0 - g_pct if total > 0 else 0.0
        return round(g, 3), round(w, 3), round(g_pct, 3), round(w_pct, 3)
    except Exception:
        return None, None, None, None

# ------------- ① Tokenizer ----------------- #
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# 固定随机种子
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# ============ 构造样本（加入 FORMAT INSTRUCTION & 结构化头） ============ #
FMT_INSTRUCTION = (
    "FORMAT INSTRUCTION:\n"
    "First output exactly ONE line:\n"
    "5G_MHz=<value>; WiFi_MHz=<value>; 5G_%=<value>; WiFi_%=<value>\n"
    "Then output the sector JSON with keys A/B/C/D as provided in training data.\n"
    "Do not add extra fields in JSON.\n"
    "----\n"
)

def build_pair(example):
    user_raw = example["prompt"].strip()
    comp_raw = example["completion"].strip()

    # 由 JSON 计算汇总头
    g5, wf, g5p, wfp = parse_totals_from_json(comp_raw)
    header = None
    if g5 is not None and wf is not None:
        header = f"5G_MHz={g5}; WiFi_MHz={wf}; 5G_%={g5p}; WiFi_%={wfp}"

    # user 侧加入格式提示
    user_with_fmt = FMT_INSTRUCTION + user_raw

    # assistant 侧把头部插到 JSON 前
    if header:
        assistant_aug = header + "\n" + comp_raw
    else:
        assistant_aug = comp_raw

    # 用 chat 模板
    messages = [
        {"role": "system", "content": "You are a spectrum allocation assistant."},
        {"role": "user", "content": user_with_fmt},
        {"role": "assistant", "content": assistant_aug}
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # 做“只对 assistant 计 loss”的 label masking
    user_prefix = tok.apply_chat_template(
        messages[:-1] + [{"role": "assistant", "content": ""}],
        tokenize=False, add_generation_prompt=False
    )
    user_ids  = tok(user_prefix, truncation=True, max_length=MAX_LEN, add_special_tokens=False)["input_ids"]
    full_ids  = tok(text,        truncation=True, max_length=MAX_LEN, add_special_tokens=False)["input_ids"]
    labels    = full_ids.copy()
    for i in range(min(len(user_ids), len(labels))):
        labels[i] = -100

    return {"input_ids": full_ids, "labels": labels}

raw_ds = load_dataset("json", data_files=DATA_PATH, split="train")
processed = raw_ds.map(build_pair, remove_columns=raw_ds.column_names)

split = processed.train_test_split(test_size=0.05, seed=SEED)
train_ds, val_ds = split["train"], split["test"]

# ============ ③ 基座模型（8bit，若不可用则回退 FP16） ============ #
model_kwargs = dict(trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
if BitsAndBytesConfig is not None:
    try:
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
        model_kwargs["quantization_config"] = bnb_cfg
    except Exception:
        pass

base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)

# 如果可用，做 k-bit 训练准备
if prepare_model_for_kbit_training is not None and "quantization_config" in model_kwargs:
    try:
        base_model = prepare_model_for_kbit_training(base_model)
    except Exception:
        pass

# 开启梯度检查点（老版本有时为 .enable()）
try:
    base_model.gradient_checkpointing_enable()
except Exception:
    try:
        base_model.gradient_checkpointing = True
    except Exception:
        pass

# ④ LoRA
lora_cfg = LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_ALPHA,
    target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
    lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_cfg)
print(model.print_trainable_parameters())

# ⑤ 自定义 collator（只负责 pad；labels 中的 -100 已准备好）
def pad_collator(features):
    max_len = max(len(f["input_ids"]) for f in features)
    input_ids, labels, attn = [], [], []
    pad_id = tok.pad_token_id
    for f in features:
        ids = f["input_ids"]
        lbs = f["labels"]
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id]*pad)
        labels.append(lbs + [-100]*pad)
        attn.append([1]*len(ids) + [0]*pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }

# ⑥ Trainer（通过兼容构造器创建 TrainingArguments）
CFG = {
    "output_dir": OUTPUT_DIR,
    "per_device_train_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRAD_ACC,
    "learning_rate": LR,
    "num_train_epochs": EPOCHS,
    "logging_steps": 50,
    "save_steps": 500,
    "save_total_limit": 2,
    "fp16": True,
    "optim": "paged_adamw_8bit",
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "report_to": "tensorboard",
    "logging_dir": "./logs",
    "max_grad_norm": 1.0,
    "seed": SEED,
    "evaluation_strategy": "steps",  # 若不被支持将自动回退
    "eval_steps": EVAL_STEPS,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
}
args = make_training_args_compatible(CFG)

trainer = Trainer(
    model         = model,
    args          = args,
    train_dataset = train_ds,
    eval_dataset  = val_ds,
    data_collator = pad_collator,
    callbacks     = [PeriodicEvalCallback(eval_steps=EVAL_STEPS)],
)

# ⑦ 自动续训
resume_ckpt = None
if os.path.isdir(OUTPUT_DIR):
    ckpts = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
    if ckpts:
        resume_ckpt = max(ckpts, key=lambda x: int(x.split("-")[-1]))
        print(f"Resume from {resume_ckpt}")

trainer.train(resume_from_checkpoint=resume_ckpt)

# ⑧ 评估 + 保存
try:
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    trainer.save_state()
except Exception:
    pass

trainer.save_model(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}")
