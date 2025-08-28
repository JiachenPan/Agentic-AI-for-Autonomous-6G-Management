# -*- coding: utf-8 -*-
"""
finetune_qwen3_from_v2_live_eval_targeted.py

在 v2 LoRA 基础上进行“对症”二次微调（v3）：
- 只对答案段计损（<answer>...</answer>），避免学到提示词和无关上下文。
- 引入“难例加权采样”：基于评测 CSV 中 band_ok 失败的样本（按 id 对齐）提升采样概率。
- 强化结构化输出：在 system 提示中明确 JSON 结构与和/比例约束（训练与推理同构）。
- 继续支持“旧版兼容”的 TrainingArguments 过滤与安全 resume。

可选（默认关闭）：
- 轻量一致性罚项（对模型当前 step 的“解码 JSON”做和/比例校验并追加很小的 penalty）。
  该项计算昂贵，默认关闭，建议小批量实验。

使用说明（关键变量在“路径/开关”处）：
- EVAL_CSV_PATH 指向本次评测 CSV（含 id 与 band_ok/base_band_ok/lora_band_ok 任一）。
- HARD_WEIGHT 控制失败样本的过采样权重（如 3~6）。
- ENABLE_CONSTRAINT_PENALTY 若要打开一致性罚项，设 True 并适当调小 lambda_c。

依赖：transformers >= 4.36，peft，datasets，bitsandbytes
"""

import os
import glob
import json
import random
from inspect import signature
from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, TrainerCallback
)
from peft import PeftModel, prepare_model_for_kbit_training

# ======================== 路径 / 开关（请按需修改） ======================== #
BASE_MODEL_PATH = "/root/autodl-tmp/Qwen/Qwen3-8B"          # 基座模型
V2_ADAPTER_DIR  = "/root/autodl-tmp/qwen3-8B-ft-v5"         # v2 LoRA 目录（从这里继续训练）
DATA_PATH       = "/root/autodl-tmp/AoA_dataset.jsonl"     # 训练数据（jsonl）
OUTPUT_DIR      = "./qwen3-8B-ft-v5-AOA"                    # 输出目录
EVAL_CSV_PATH   = "/root/autodl-tmp/AOA_eval_compare-Qwen3-new.csv"  # 评测表（用于难例加权）

# ========================== 训练超参 ========================== #
BATCH_SIZE = 1
GRAD_ACC   = 8
LR         = 8e-5             # 建议比原来更“温和”的学习率范围：5e-5 ~ 1e-4
EPOCHS     = 2
MAX_LEN    = 2048
SEED       = 42

LOG_STEPS  = 50
SAVE_STEPS = 500
SAVE_LIMIT = 2
EVAL_EVERY = SAVE_STEPS        # 每隔多少步在终端打印一次 eval loss

# 难例加权
HARD_WEIGHT = 4.0              # band_ok=0 的样本采样权重
BASE_WEIGHT = 1.0

# 一致性罚项（可选）
ENABLE_CONSTRAINT_PENALTY = False
LAMBDA_CONSTRAINT = 0.1        # 罚项权重，开启时建议 0.05~0.2

# JSON 答案标记（用于“只对答案计损”）
ANSWER_START = "<answer>"
ANSWER_END   = "</answer>"

# ========================== 实用函数 ========================== #
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def assert_exists(path, what):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{what} not found: {path}")

def torch_lt_26() -> bool:
    v = torch.__version__.split("+")[0].strip()
    parts = v.split(".")
    try:
        major = int(parts[0]); minor = int(parts[1])
    except Exception:
        return True
    return (major < 2) or (major == 2 and minor < 6)

def make_training_args(**kwargs) -> TrainingArguments:
    """仅保留当前 transformers 版本支持的 TrainingArguments 参数，避免旧版因未知参数报错。"""
    sig = set(signature(TrainingArguments.__init__).parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in sig}
    return TrainingArguments(**filtered)

# ========================== 前置检查 ========================== #
assert_exists(BASE_MODEL_PATH, "BASE_MODEL_PATH")
assert_exists(V2_ADAPTER_DIR,  "V2_ADAPTER_DIR")
assert_exists(DATA_PATH,       "DATA_PATH")
os.environ.setdefault("WANDB_DISABLED", "true")
set_seed(SEED)

# ========================== Tokenizer ========================== #
tok = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=False
)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# 添加答案标记，便于“只对答案计损”
added = {"additional_special_tokens": [ANSWER_START, ANSWER_END]}
num_added = tok.add_special_tokens(added)

# ====================== 数据集：模板 + 只对答案计损 ====================== #
ROLE_MAP = {"user": "user", "assistant": "assistant", "system": "system"}

SYSTEM_JSON_INSTRUCT = (
    "你是射频制式抽取器。请以严格 JSON 输出，不要任何多余文本。要求：\n"
    "1) g5_mhz + wifi_mhz == total_mhz; 2) used_g5_pct + used_wifi_pct == 100;\n"
    "3) 字段: {\"band_ok\": bool, \"g5_mhz\": number, \"wifi_mhz\": number, \"total_mhz\": number,\n"
    "          \"used_g5_pct\": number, \"used_wifi_pct\": number}；单位统一为 MHz，百分比 0~100。\n"
    "只输出 JSON。"
)

def wrap_messages_with_json_system(conv: List[Dict[str, str]]) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_JSON_INSTRUCT}]
    for t in conv:
        role = ROLE_MAP.get(str(t.get("from", "")).lower())
        if role not in ("user", "assistant", "system"):
            continue
        content = (t.get("value") or "").strip()
        if not content:
            continue
        if role == "assistant":
            # 仅在 assistant 段落外包裹 <answer>...
            content = f"{ANSWER_START}{content}{ANSWER_END}"
        messages.append({"role": role, "content": content})
    if len(messages) == 1:  # 只有 system
        messages.append({"role": "user", "content": ""})
    return messages


def apply_template(example: Dict[str, Any]) -> Dict[str, Any]:
    conv = example.get("conversations", [])
    messages = wrap_messages_with_json_system(conv)
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # 保留 id 以便采样加权
    ex = {"text": text + tok.eos_token}
    if "id" in example:
        ex["id"] = example["id"]
    return ex

raw_ds = load_dataset("json", data_files=DATA_PATH, split="train")
processed = raw_ds.map(apply_template, remove_columns=[c for c in raw_ds.column_names if c != "id"])  # 只保留 id

# 先 tokenize，再由 collator 统一 padding + 构造 labels（只对答案计损）
def tokenize(examples):
    return tok(examples["text"], truncation=True, max_length=MAX_LEN, padding=False)

processed = processed.map(tokenize, batched=True, num_proc=max(os.cpu_count() or 2, 2))
split = processed.train_test_split(test_size=0.05, seed=SEED)
train_ds, val_ds = split["train"], split["test"]

# ============================ 基座（8bit） ============================ #
bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_cfg,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto",
    local_files_only=True,
)
base_model = prepare_model_for_kbit_training(base_model)
# 由于添加了额外 token，需要 resize embedding
try:
    base_model.resize_token_embeddings(len(tok))
except Exception:
    pass

try:
    base_model.gradient_checkpointing_enable()
except Exception:
    pass

# ====================== 挂载 v2 LoRA 并设为可训练 ====================== #
model = PeftModel.from_pretrained(base_model, V2_ADAPTER_DIR, is_trainable=True)
try:
    print(model.print_trainable_parameters())
except Exception:
    pass

# ===================== 自定义 Collator：只对答案段计损 ===================== #
class DataCollatorForAnswerOnly:
    def __init__(self, tokenizer, answer_start: str, answer_end: str):
        self.tok = tokenizer
        # 获取标记对应 id（作为 added special tokens）
        self.start_id = self.tok.convert_tokens_to_ids(answer_start)
        self.end_id   = self.tok.convert_tokens_to_ids(answer_end)
        assert self.start_id != self.tok.unk_token_id and self.end_id != self.tok.unk_token_id, \
            "请确保 <answer> 与 </answer> 被加入 tokenizer 的 additional_special_tokens。"

    def __call__(self, features: List[Dict[str, Any]]):
        # transformers 的默认 collate 会 pad；我们手动做
        batch = {}
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        pad_id = self.tok.pad_token_id
        for f in features:
            ids = f["input_ids"]
            mask = f.get("attention_mask", [1]*len(ids))
            # padding
            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids = ids + [pad_id]*pad_len
                mask = mask + [0]*pad_len
            # 构造 labels：默认 -100（不计损），答案段内保留 token id
            lbl = [-100]*max_len
            # 查找所有 <answer>...</answer> 区间
            i = 0
            while i < max_len:
                if ids[i] == self.start_id:
                    j = i + 1
                    while j < max_len and ids[j] != self.end_id:
                        # 答案 token 计损
                        lbl[j] = ids[j]
                        j += 1
                    # end 标记本身不计损
                    i = j + 1
                else:
                    i += 1
            input_ids.append(ids)
            attention_mask.append(mask)
            labels.append(lbl)
        batch["input_ids"] = torch.tensor(input_ids, dtype=torch.long)
        batch["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        # 透传 id（用于采样器权重）
        if "id" in features[0]:
            batch["sample_ids"] = [f.get("id") for f in features]
        return batch

collator = DataCollatorForAnswerOnly(tok, ANSWER_START, ANSWER_END)

# ====================== 难例加权：从评测表读取 band_ok ====================== #
def build_sample_weights(train_dataset, eval_csv_path: str, hard_weight=HARD_WEIGHT, base_weight=BASE_WEIGHT):
    if not os.path.exists(eval_csv_path):
        print(f"[hard_sampling] 未找到评测表: {eval_csv_path}，将不启用难例加权。")
        return None
    import pandas as pd
    df = pd.read_csv(eval_csv_path)
    col_ok = None
    for c in ("lora_band_ok", "base_band_ok", "band_ok"):
        if c in df.columns:
            col_ok = c; break
    if col_ok is None:
        print("[hard_sampling] 评测表中未找到 band_ok 列，跳过难例加权。")
        return None
    # 统一转为 0/1
    def to01(x):
        s = str(x).strip().lower()
        if s in ("1","true","yes","y"): return 1.0
        if s in ("0","false","no","n"): return 0.0
        try:
            v = float(s); return 1.0 if v >= 0.5 else 0.0
        except:
            return np.nan
    m = df.set_index("id")[col_ok].map(to01)

    ids = train_dataset["id"] if "id" in train_dataset.column_names else [None]*len(train_dataset)
    weights = []
    miss = 0
    for _id in ids:
        if _id in m.index:
            w = hard_weight if m.loc[_id] == 0.0 else base_weight
        else:
            w = base_weight; miss += 1
        weights.append(w)
    print(f"[hard_sampling] 失败样本权重={hard_weight}，普通样本权重={base_weight}，未对齐 id 数={miss}。")
    return torch.tensor(weights, dtype=torch.float)

train_weights = build_sample_weights(train_ds, EVAL_CSV_PATH, HARD_WEIGHT, BASE_WEIGHT)

# ============================ 自定义 Trainer ============================ #
class WeightedTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            return super().get_train_dataloader()
        # 分布式/TPU 等场景，退回默认采样器（避免不兼容）
        is_dist = getattr(self.args, "local_rank", -1) != -1 or getattr(self.args, "world_size", 1) > 1
        if train_weights is None or is_dist:
            return super().get_train_dataloader()
        sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"), labels=labels)
        loss = outputs.loss
        # 可选：一致性罚项（对当前 step 的贪婪解码 JSON 做和/比例校验）
        if ENABLE_CONSTRAINT_PENALTY and LAMBDA_CONSTRAINT > 0:
            try:
                with torch.no_grad():
                    # 仅解码被计损的片段，减少无关 token
                    mask = (labels != -100)
                    pred_ids = torch.argmax(outputs.logits, dim=-1)
                    # 只取每个样本第一段答案的 token 进行解析
                    penalties = []
                    for b in range(pred_ids.size(0)):
                        idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
                        if idx.numel() == 0:
                            penalties.append(0.0); continue
                        # 裁一小段，避免过长解码开销
                        sl = idx.min().item(); sr = idx.max().item() + 1
                        sl = max(0, sl - 8); sr = min(pred_ids.size(1), sr + 4)
                        text = tok.decode(pred_ids[b, sl:sr], skip_special_tokens=True)
                        penalties.append(float(_json_constraint_penalty(text)))
                if len(penalties) > 0:
                    penalty_tensor = torch.tensor(penalties, dtype=loss.dtype, device=loss.device)
                    loss = loss + LAMBDA_CONSTRAINT * penalty_tensor.mean()
            except Exception:
                pass
        return (loss, outputs) if return_outputs else loss

# 轻量 JSON 约束罚项：只对 sum/百分比等进行 L1 惩罚
import re as _re

def _safe_float(x):
    try:
        return float(x)
    except:
        return None

def _json_constraint_penalty(text: str) -> float:
    # 尝试读 JSON，否则从文本里粗抽数
    d = None
    try:
        d = json.loads(text)
    except Exception:
        # 粗匹配：g5_mhz: 123 等
        pairs = {
            "g5_mhz": _re.search(r"g5_mhz\D+([0-9]+(?:\.[0-9]+)?)", text or ""),
            "wifi_mhz": _re.search(r"wifi_mhz\D+([0-9]+(?:\.[0-9]+)?)", text or ""),
            "total_mhz": _re.search(r"total_mhz\D+([0-9]+(?:\.[0-9]+)?)", text or ""),
            "used_g5_pct": _re.search(r"used_g5_pct\D+([0-9]+(?:\.[0-9]+)?)", text or ""),
            "used_wifi_pct": _re.search(r"used_wifi_pct\D+([0-9]+(?:\.[0-9]+)?)", text or ""),
        }
        d = {k: (_safe_float(m.group(1)) if m else None) for k, m in pairs.items()}
    if not isinstance(d, dict):
        return 0.0
    p = 0.0
    a = d.get("g5_mhz"); b = d.get("wifi_mhz"); t = d.get("total_mhz")
    if None not in (a,b,t):
        p += abs((a + b) - t)
    g = d.get("used_g5_pct"); w = d.get("used_wifi_pct")
    if None not in (g,w):
        p += abs((g + w) - 100.0)
    return float(p)

# ============================ 训练参数 ============================ #
args = make_training_args(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    logging_steps=LOG_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_LIMIT,
    fp16=True,
    optim="paged_adamw_8bit",      # 若旧版不支持会被过滤
    lr_scheduler_type="cosine",    # 同上
    warmup_ratio=0.05,
    report_to="tensorboard",
    logging_dir="./logs",
    max_grad_norm=1.0,
    seed=SEED,
    save_safetensors=True,          # 旧版没有会被过滤
    overwrite_output_dir=True,      # 旧版没有会被过滤
)

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,          # 手动 evaluate
    data_collator=collator,
    tokenizer=tok,
)

# ===================== 回调：定期评估并打印 ===================== #
class LiveEvalPrinter(TrainerCallback):
    def __init__(self, trainer_obj, every_steps=EVAL_EVERY):
        self.trainer = trainer_obj
        self.every = max(1, int(every_steps))

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.every == 0:
            try:
                metrics = self.trainer.evaluate()
                ev = metrics.get("eval_loss")
                es = metrics.get("eval_samples")
                print(f"\n[Eval] step={state.global_step}  eval_loss={ev:.6f}  (samples={es})\n", flush=True)
                try:
                    self.trainer.log(metrics)
                    self.trainer.save_metrics("eval", metrics)
                except Exception:
                    pass
            except Exception as e:
                print(f"[Eval callback error] {e}", flush=True)
        return control

trainer.add_callback(LiveEvalPrinter(trainer, every_steps=EVAL_EVERY))

# =================== 续训：根据 torch 版本安全处理 =================== #
resume_ckpt = None
if os.path.isdir(OUTPUT_DIR):
    ckpts = [p for p in glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")) if os.path.isdir(p)]
    if ckpts:
        last_ckpt = None
        try:
            last_ckpt = max(ckpts, key=lambda x: int(x.split("-")[-1]))
        except Exception:
            last_ckpt = sorted(ckpts)[-1]
        if torch_lt_26():
            print(f"[WARN] torch {torch.__version__} < 2.6，因安全限制将跳过从 checkpoint 恢复。需要续训请手动升级 torch。")
            resume_ckpt = None
        else:
            print("Resuming from", last_ckpt)
            resume_ckpt = last_ckpt

trainer.train(resume_from_checkpoint=resume_ckpt)

# ========================== 结束评估 + 保存 ========================== #
try:
    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)
    trainer.log_metrics("eval_final", metrics)
    trainer.save_metrics("eval_final", metrics)
except Exception as e:
    print("[WARN] final evaluate() failed:", e)

trainer.save_state()
trainer.save_model(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print("Done. LoRA v3 saved to", OUTPUT_DIR)
