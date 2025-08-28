"""
Qwen3-8B + QLoRA (4-bit) SFT — masks loss to assistant-only tokens, injects a strict-JSON system prompt,
adds quick post-train structural eval (JSON validity & keys), and improves training knobs.

Key changes vs v2:
1) Switch to QLoRA (4-bit NF4) for better adaptation & memory efficiency.
2) Correct SFT loss: only the assistant portion contributes to loss (no training on user text).
3) Inject a minimal system prompt that enforces EXACT JSON shape.
4) Slightly stronger LR & regularization; bf16 when available; gradient checkpointing.
5) Default data collator (no MLM) + remove_unused_columns=False.
6) Optional quick_eval() at the end to measure JSON validity and key presence on val split.
7) Same auto-resume & steady-decrease check retained.
"""
import os, glob, re, json, random
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, TrainerCallback
)
# 某些较老版本 transformers 没有 default_data_collator，这里做兼容导入
try:
    from transformers import default_data_collator
except Exception:
    default_data_collator = None
from dataclasses import fields as _da_fields
from packaging import version as _v
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ----------------- 超参 ----------------- #
MODEL_PATH = "/root/autodl-tmp/Qwen/Qwen3-8B"
DATA_PATH  = "FCB_dataset.jsonl"
OUTPUT_DIR = "./qwen3-8B-ft-v5"
BATCH_SIZE = 1
GRAD_ACC   = 16
LR         = 2e-4               # 更适合 LoRA 的学习率
EPOCHS     = 4                  # 收敛更稳，结合更强 LR
LORA_RANK  = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
MAX_LEN    = 2048
EVAL_STEPS = 200
SEED       = 42

# ------------- Callback：定期评估 ------------ #
class PeriodicEvalCallback(TrainerCallback):
    def __init__(self, eval_steps=500):
        self.eval_steps = eval_steps
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            control.should_evaluate = True
        return control

# ----------------- 实用函数 ----------------- #
# 解析 RSSI 文本 -> 计算每扇区总和与相对强度（与 v2 保持一致）
rssi_pat = re.compile(r'([ABCD])_([a-fA-F]):\s*([0-9]+(?:\.[0-9]+)?)')

def extract_sector_features(text: str):
    sums = {s: 0.0 for s in "ABCD"}
    for m in rssi_pat.finditer(text):
        sector = m.group(1).upper()
        val = float(m.group(3))
        sums[sector] += abs(val)
    max_sum = max(sums.values()) if sums else 1.0
    rel = {k: (sums[k] / max_sum if max_sum > 0 else 0.0) for k in sums}
    return sums, rel

def features_str_from_prompt(user_text: str) -> str:
    sums, rel = extract_sector_features(user_text)
    return (
        f"[FEATS] sum:A={sums['A']:.3f},B={sums['B']:.3f},C={sums['C']:.3f},D={sums['D']:.3f}; "
        f"rel:A={rel['A']:.4f},B={rel['B']:.4f},C={rel['C']:.4f},D={rel['D']:.4f}"
    )

# ---- 强制 JSON 的系统提示（可按需调整文案，但保持稳定） ---- #
SYSTEM_PROMPT = (
    "You are an RF allocation assistant. Return STRICT JSON only. "
    "No prose, no comments. The JSON MUST have EXACTLY two top-level keys: \"NR-U\" and \"WiFi6\". "
    "All values are MHz integers or floats."
)

# ------------- ① Tokenizer ----------------- #
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# Qwen3 通常无 pad，沿用 EOS
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
# 右侧补齐，有利于生成任务
try:
    tok.padding_side = "right"
except Exception:
    pass

# 固定随机种子
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# ------------- ② 数据集（正确的 SFT 标签掩码） --------- #
# 思路：先用 chat template 编出 prompt_ids（到 assistant 起始），再编出 full_ids（含助手完整回答）。
# labels 对 prompt 部分置 -100，只对助手输出计算 loss。

def build_messages(example):
    user = example["prompt"]
    feats = features_str_from_prompt(user)
    user_with_feats = user.strip() + "\n" + feats + "\n"
    assistant = example["completion"].strip()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_with_feats},
        {"role": "assistant", "content": assistant},
    ]
    return messages


def encode_with_masks(example):
    messages = build_messages(example)
    # prompt：到 assistant 开始（不含助手内容）
    prompt_ids = tok.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        continue_final_message=False,  # 保险：以 assistant 起始处结束
    )
    # full：包含助手完整回答
    full_ids = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    # 截断
    input_ids = full_ids[:MAX_LEN]
    prompt_len = min(len(prompt_ids), len(input_ids))
    labels = input_ids.copy()
    # 只训练助手部分
    for i in range(prompt_len):
        labels[i] = -100
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        # 用于按长度分桶（可选）
        "length": len(input_ids),
    }

raw_ds = load_dataset("json", data_files=DATA_PATH, split="train")
# 先划分原始集，以便后面做 quick_eval
raw_split = raw_ds.train_test_split(test_size=0.05, seed=SEED)
raw_train, raw_val = raw_split["train"], raw_split["test"]

# 编码 + 掩码
def _num_proc_for(ds):
    try:
        n = len(ds)
    except Exception:
        n = 2
    cpu = os.cpu_count() or 2
    return max(1, min(cpu, n))

processed_train = raw_train.map(encode_with_masks, remove_columns=raw_train.column_names,
                                num_proc=_num_proc_for(raw_train))
processed_val   = raw_val.map(encode_with_masks,   remove_columns=raw_val.column_names,
                              num_proc=_num_proc_for(raw_val))

# ------------- ③ 基座模型（QLoRA 4-bit） ----------- #
# bfloat16 可用时优先
bf16_available = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
compute_dtype = torch.bfloat16 if bf16_available else torch.float16

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_cfg,
    torch_dtype=compute_dtype,
    trust_remote_code=True,
    device_map="auto",
)

# 训练前必要准备
base_model = prepare_model_for_kbit_training(base_model)
base_model.gradient_checkpointing_enable()
base_model.config.use_cache = False  # 否则与 checkpointing 冲突

# ------------- ④ LoRA ---------------------- #
lora_cfg = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "up_proj","down_proj","gate_proj",
    ],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_cfg)
print(model.print_trainable_parameters())

# ------------- ⑤ Trainer 参数 --------------- #
# 兼容不同 transformers 版本：仅传入该版本支持的参数；若不支持 evaluation_strategy，则不在训练中自动评估（我们仍会在训练后手动评估）
def make_training_args(**kwargs):
    allowed = {f.name for f in _da_fields(TrainingArguments)}
    supports_eval_strategy = ("evaluation_strategy" in allowed)
    supports_save_strategy = ("save_strategy" in allowed)

    # 旧版兼容：没有 evaluation_strategy 时，去掉相关参数并禁用 load_best_model_at_end
    if not supports_eval_strategy:
        kwargs.pop("evaluation_strategy", None)
        kwargs.pop("metric_for_best_model", None)
        kwargs.pop("greater_is_better", None)
        if ("evaluate_during_training" in allowed and kwargs.get("eval_steps")):
            kwargs["evaluate_during_training"] = True
        kwargs["load_best_model_at_end"] = False

    # 新版：确保 save/eval strategy 对齐到 steps（若字段存在）
    if supports_eval_strategy:
        kwargs.setdefault("evaluation_strategy", "steps")
    if supports_save_strategy:
        kwargs.setdefault("save_strategy", "steps")

    pruned = {k: v for k, v in kwargs.items() if k in allowed}
    return TrainingArguments(**pruned)

args = make_training_args(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=max(1, BATCH_SIZE),
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    logging_steps=50,
    save_steps=1000,
    save_total_limit=3,
    bf16=bf16_available,
    fp16=not bf16_available,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.05,
    report_to="tensorboard",
    logging_dir="./logs",
    max_grad_norm=0.3,
    seed=SEED,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_checkpointing=True,
    group_by_length=True,
    dataloader_num_workers=max(2, (os.cpu_count() or 2) // 2),
    remove_unused_columns=False,  # 重要：不要丢掉 labels
    save_safetensors=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=processed_train,
    eval_dataset=processed_val,
    data_collator=(default_data_collator if default_data_collator is not None else None),
    callbacks=[PeriodicEvalCallback(eval_steps=EVAL_STEPS)],
)

# ------------- ⑥ 自动续训（兼容 torch<2.6 的安全限制；必要时仅加载权重） ------------------- #
resume_ckpt = None
latest_ckpt = None
if os.path.isdir(OUTPUT_DIR):
    ckpts = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
    if ckpts:
        latest_ckpt = max(ckpts, key=lambda x: int(x.split("-")[-1]))
        try:
            allow_resume = _v.parse(torch.__version__.split("+")[0]) >= _v.parse("2.6.0")
        except Exception:
            allow_resume = False
        if allow_resume:
            resume_ckpt = latest_ckpt
            print(f"⏪ Resuming from {resume_ckpt}")
        else:
            print("⛔️ Skip resume: torch<2.6 禁止加载 optimizer 状态（CVE-2025-32434）。改为仅加载 LoRA 权重，不恢复优化器。")
    else:
        print("⚠️  If no checkpoint is detected, training will start from scratch.")

# torch<2.6：不恢复优化器，但可以把 LoRA 权重加载进来
if latest_ckpt and resume_ckpt is None:
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, latest_ckpt)
        print(f"🔁 Weights-only loaded from {latest_ckpt}")
    except Exception as e:
        print(f"[WARN] Weights-only load failed: {e}")

trainer.train(resume_from_checkpoint=resume_ckpt)

# ------------- ⑦ 评估 + 保存（确保写出 trainer_state.json） ------- #
eval_metrics = trainer.evaluate()
trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)
trainer.save_state()

# ------------- ⑧ 保存权重 ----------------------- #
trainer.save_model(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print(f"\n🍻 Fine-tuning complete! LoRA + 4-bit weights have been saved to {OUTPUT_DIR}\n")

# ------------- ⑨ 训练后：检查 eval_loss 是否稳降 ------------- #
def steadily_down(evals, rel_drop=0.01, k=2):
    cnt = 0
    for (_, a), (_, b) in zip(evals, evals[1:]):
        if b <= a * (1 - rel_drop):
            cnt += 1
    return cnt >= k

ts_path = os.path.join(OUTPUT_DIR, "trainer_state.json")
if os.path.isfile(ts_path):
    try:
        hist = json.load(open(ts_path))["log_history"]
        evals = [(x.get("step", -1), x["eval_loss"]) for x in hist if "eval_loss" in x]
        print("\n========== Last 10 times eval_loss ==========")
        for s, l in evals[-10:]:
            print(f"step={s:<8} eval_loss={l:.6f}")
        print("\nSteadily decreasing:", steadily_down(evals))
        print("==========================================\n")
    except Exception as e:
        print(f"[WARN] Read/parse {ts_path} Failed: {e}")
else:
    print(f"[WARN] {ts_path} not found; training may not have completed or the state file wasn’t written.")

# ------------- ⑩ Quick structural eval on val set (JSON&keys) ------------- #
# 这一步不会影响训练；用于快速判断“是否输出有效 JSON 且包含所需键”。
# 若你用自定义评测脚本，可忽略此块。

def quick_eval(max_samples=50, max_new_tokens=192):
    model.eval()
    n = min(max_samples, len(raw_val))
    ok_json = 0
    ok_keys = 0
    needed_keys = {"NR-U", "WiFi6"}

    for i in range(n):
        ex = raw_val[i]
        # 仅到 assistant 开头，促使模型补全回答
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["prompt"].strip() + "\n" + features_str_from_prompt(ex["prompt"]) + "\n"},
        ]
        input_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        input_ids = torch.tensor([input_ids]).to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.2,
                top_p=0.9,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
        text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        # 解析 JSON
        is_json = False
        has_keys = False
        try:
            obj = json.loads(text)
            is_json = True
            has_keys = set(obj.keys()) == needed_keys or needed_keys.issubset(set(obj.keys()))
        except Exception:
            # 尝试截断到最近的 '}'
            if '}' in text:
                try:
                    cand = text.split('}', 1)[0] + '}'
                    obj = json.loads(cand)
                    is_json = True
                    has_keys = set(obj.keys()) == needed_keys or needed_keys.issubset(set(obj.keys()))
                except Exception:
                    pass
        ok_json += int(is_json)
        ok_keys += int(has_keys)
    print(f"[quick_eval] JSON ok: {ok_json}/{n} | Keys ok: {ok_keys}/{n}")

try:
    quick_eval()
except Exception as e:
    print(f"[quick_eval] skipped due to error: {e}")

