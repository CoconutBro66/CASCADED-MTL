# sft_T2_fine.py
import os
import json
from typing import Dict, Any

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from peft import LoraConfig, get_peft_model, PeftModel

# ====================
# Config
# ====================
# model_path = "/media/qust521/92afc6a9-13fd-4458-8a46-4b008127de08/LLMs/Qwen2-7B-Instruct"
model_path = "/media/qust521/92afc6a9-13fd-4458-8a46-4b008127de08/LLMs/Meta-Llama3-8B-Instruct"
train_file = "/home/qust521/Projects/GATE_NLP/Abusive_L2D/target_detection/LLMs/Mutli_Task/train_data/train.jsonl"
dev_file   = "/home/qust521/Projects/GATE_NLP/Abusive_L2D/target_detection/LLMs/Mutli_Task/train_data/dev.jsonl"
test_file  = "/home/qust521/Projects/GATE_NLP/Abusive_L2D/target_detection/LLMs/Mutli_Task/train_data/test.jsonl"

output_dir = "./result_T2_Meta-Llama3-8B-Instruct"
max_seq_len = 256
num_train_epochs = 10
learning_rate = 2e-5
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 8

VALID_FINE_LABELS = [
    "death_threat", "sexual_assault", "sexual_explicit", "physical_harm",
    "radiation_of_threats", "attacks_on_credibility", "misogynistic",
    "homophobic", "religious", "political_sectarian", "racist", "general"
]
FINE_LABEL2ID = {l: str(i) for i, l in enumerate(VALID_FINE_LABELS)}
VALID_FINE_IDS = [str(i) for i in range(len(VALID_FINE_LABELS))]

FINE_ID_TEXT = "\n".join([f"{i} = {lab}" for i, lab in enumerate(VALID_FINE_LABELS)])

T2_INSTRUCTION = f"""You are a content review expert.

Task: Given a text confirmed to be targeted abusive. output ONLY the fine-grained category ID.

Choose exactly ONE ID from this closed set:
{FINE_ID_TEXT}

Output ONLY the ID (0-11). No explanation.
"""

# ====================
# Load JSONL
# ====================
def load_raw_jsonl(path: str) -> Dataset:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} samples from {path}")
    return Dataset.from_list(data)

train_raw = load_raw_jsonl(train_file)
dev_raw   = load_raw_jsonl(dev_file)
test_raw  = load_raw_jsonl(test_file)

# filter targeted-abusive + valid fine_label
def keep_t2(x):
    return (x.get("tri_label", "") == "targeted-abusive") and (x.get("fine_label", None) in VALID_FINE_LABELS)

train_raw = train_raw.filter(keep_t2)
dev_raw   = dev_raw.filter(keep_t2)
test_raw  = test_raw.filter(keep_t2)



print(f"After filtering: train={len(train_raw)}, dev={len(dev_raw)}, test={len(test_raw)}")

# ====================
# Build prompt/answer
# ====================
def build_io(sample: Dict[str, Any]) -> Dict[str, str]:
    text = sample.get("text", "")
    fine_label = sample.get("fine_label", None)
    fine_id = FINE_LABEL2ID.get(fine_label, "PARSE_ERROR")

    prompt = (
        T2_INSTRUCTION + "\n\n"
        + f"Text: {text}\n"
        + "Answer:\n"
    )
    answer = fine_id
    return {"prompt": prompt, "answer": answer}

def add_prompt_answer(batch):
    prompts, answers = [], []
    for i in range(len(batch["id"])):
        s = {k: batch[k][i] for k in batch}
        io = build_io(s)
        prompts.append(io["prompt"])
        answers.append(io["answer"])
    return {"prompt": prompts, "answer": answers}

train_io = train_raw.map(add_prompt_answer, batched=True)
dev_io   = dev_raw.map(add_prompt_answer, batched=True)
test_io  = test_raw.map(add_prompt_answer, batched=True)

# ====================
# Tokenizer + tokenize
# ====================
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    prompt = example["prompt"]
    answer = example["answer"] + (tokenizer.eos_token or "")

    prompt_enc = tokenizer(prompt, truncation=True, max_length=max_seq_len, add_special_tokens=True)
    full_enc = tokenizer(prompt + answer, truncation=True, max_length=max_seq_len,
                         padding="max_length", add_special_tokens=True)

    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]
    labels = input_ids.copy()

    prompt_len = len(prompt_enc["input_ids"])
    prompt_len = min(prompt_len, len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_tok = train_io.map(tokenize, remove_columns=train_io.column_names, desc="Tokenizing train")
dev_tok   = dev_io.map(tokenize, remove_columns=dev_io.column_names, desc="Tokenizing dev")
test_tok  = test_io.map(tokenize, remove_columns=test_io.column_names, desc="Tokenizing test")

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ====================
# QLoRA (4bit) + LoRA
# ====================


base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    # 针对 Qwen2 系列模型，建议注入以下模块以获得最佳微调效果
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# ====================
# Parse + Eval
# ====================
def safe_parse_t2_id(ans_text: str) -> str:
    s = (ans_text or "").strip().lstrip()
    # 取前面的整数（可能输出 "11"）
    num = ""
    for ch in s:
        if ch.isdigit():
            num += ch
        else:
            break
    if num in VALID_FINE_IDS:
        return num
    return "PARSE_ERROR"

@torch.no_grad()
def evaluate_t2(model, dataset_io, desc="dev"):
    model.eval()
    y_true, y_pred = [], []
    parse_err = 0

    for i in range(len(dataset_io)):
        sample = dataset_io[i]
        prompt = sample["prompt"]
        gold = (sample["answer"] or "").strip()
        y_true.append(gold)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            temperature=0.0
        )
        gen = gen_ids[0][inputs.input_ids.shape[1]:]
        out = tokenizer.decode(gen, skip_special_tokens=True).strip()
        pred = safe_parse_t2_id(out)
        y_pred.append(pred)
        if pred == "PARSE_ERROR":
            parse_err += 1

    # fine_acc = accuracy_score(y_true, y_pred)
    _, _, fine_f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=VALID_FINE_IDS, average="macro", zero_division=0
    )

    metrics = {
        # f"{desc}_fine_accuracy": fine_acc,
        f"{desc}_fine_f1_macro": fine_f1_macro,
        f"{desc}_parse_error_rate": (parse_err / max(1, len(y_true))),
    }

    print(f"\n===== {desc} evaluation (T2) =====")
    for k, v in metrics.items():
        if "rate" in k:
            print(f"{k}: {v:.4%}")
        else:
            print(f"{k}: {v:.4f}")
    print("=================================\n")
    return metrics

# ====================
# Callback: save best by dev_fine_f1_macro
# ====================
class T2EvalCallback(TrainerCallback):
    def __init__(self, dev_io, out_dir):
        self.dev_io = dev_io
        self.out_dir = out_dir
        self.best = -1.0

    def on_epoch_end(self, args, state, control, **kwargs):
        m = kwargs["model"]
        metrics = evaluate_t2(m, self.dev_io, desc="dev")
        score = metrics["dev_fine_f1_macro"]
        if score > self.best:
            self.best = score
            best_dir = os.path.join(self.out_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            print(f"New best dev_fine_f1_macro={score:.4f}, saving LoRA to {best_dir}")
            m.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

# ====================
# TrainingArguments + Trainer
# ====================
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    logging_steps=50,
    save_strategy="no",
    evaluation_strategy="no",
    bf16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[T2EvalCallback(dev_io, output_dir)]
)

# ====================
# Main
# ====================
if __name__ == "__main__":
    print("===== Start QLoRA SFT: T2 =====")
    trainer.train()
    print("===== Training done =====")

    best_model_dir = os.path.join(output_dir, "best_model")
    if os.path.exists(best_model_dir):
        print(f"Loading best adapter: {best_model_dir}")
        base_model_test = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        best_model_for_test = PeftModel.from_pretrained(base_model_test, best_model_dir)
    else:
        best_model_for_test = model

    print("===== Test evaluation (T2) =====")
    test_metrics = evaluate_t2(best_model_for_test, test_io, desc="test")

    # save test predictions
    results = []
    best_model_for_test.eval()
    for i in range(len(test_io)):
        sample = test_io[i]
        prompt = sample["prompt"]
        gold = (sample["answer"] or "").strip()

        inputs = tokenizer(prompt, return_tensors="pt").to(best_model_for_test.device)
        with torch.no_grad():
            gen_ids = best_model_for_test.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                temperature=0.0
            )
        gen = gen_ids[0][inputs.input_ids.shape[1]:]
        out = tokenizer.decode(gen, skip_special_tokens=True).strip()
        pred = safe_parse_t2_id(out)

        results.append({
            "id": test_raw[i]["id"],
            "text": test_raw[i]["text"],
            "gold": gold,
            "prediction": pred,
            "raw_output": out
        })

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "test_predictions.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(os.path.join(output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    print("Saved predictions and metrics!")
