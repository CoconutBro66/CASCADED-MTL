# sft_T3_loc.py
import os
import json
from typing import Dict, Any, List, Tuple

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
from sklearn.metrics import precision_recall_fscore_support
from peft import LoraConfig, get_peft_model, PeftModel

# ====================
# Config
# ====================
# model_path = "/media/qust521/92afc6a9-13fd-4458-8a46-4b008127de08/LLMs/Qwen2-7B-Instruct"
# model_path = "/media/qust521/92afc6a9-13fd-4458-8a46-4b008127de08/LLMs/Meta-Llama3-8B-Instruct"
model_path="/media/qust521/92afc6a9-13fd-4458-8a46-4b008127de08/LLMs/Phi-3-mini-128k-instruct"
train_file = "/home/qust521/Projects/GATE_NLP/Abusive_L2D/target_detection/LLMs/Mutli_Task/train_data/train.jsonl"
dev_file   = "/home/qust521/Projects/GATE_NLP/Abusive_L2D/target_detection/LLMs/Mutli_Task/train_data/dev.jsonl"
test_file  = "/home/qust521/Projects/GATE_NLP/Abusive_L2D/target_detection/LLMs/Mutli_Task/train_data/test.jsonl"

output_dir = "./result_T3_Phi-3-mini-128k-instruct"
max_seq_len = 512
num_train_epochs = 10
learning_rate = 2e-5
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 8

T3_INSTRUCTION = """You are a content review expert.

Task: Extract all hateful/abusive phrases from the text.

Output ONLY a JSON array (list) of strings, e.g.:
["phrase1", "phrase2"]

Rules:
- Each phrase MUST appear verbatim in the text.
- If none, output [].
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

# filter targeted-abusive only
def keep_t3(x):
    return x.get("tri_label", "") == "targeted-abusive"

train_raw = train_raw.filter(keep_t3)
dev_raw   = dev_raw.filter(keep_t3)
test_raw  = test_raw.filter(keep_t3)




print(f"After filtering: train={len(train_raw)}, dev={len(dev_raw)}, test={len(test_raw)}")

# ====================
# Build prompt/answer
# ====================
def build_io(sample: Dict[str, Any]) -> Dict[str, str]:
    text = sample.get("text", "")
    hate_phrases = sample.get("hate_phrases", [])

    spans = []
    if isinstance(hate_phrases, list):
        for p in hate_phrases:
            if isinstance(p, dict) and isinstance(p.get("content", ""), str):
                c = p["content"].strip()
                if c:
                    spans.append(c)

    prompt = (
        T3_INSTRUCTION + "\n\n"
        + f"Text: {text}\n"
        + "Answer:\n"
    )
    answer = json.dumps(spans, ensure_ascii=False)
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
# Loc helpers (token-level)
# ====================
def parse_char_pos(char_pos: str) -> Tuple[int, int]:
    try:
        s, e = char_pos.split("-")
        s, e = int(s), int(e)
        if e < s:
            e = s
        return s, e
    except:
        return 0, 0

def build_gold_token_labels(text: str, hate_phrases: List[Dict[str, Any]]) -> List[int]:
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_len,
        return_offsets_mapping=True,
        add_special_tokens=True
    )
    offsets = enc["offset_mapping"]
    labels = [0] * len(offsets)

    if not isinstance(hate_phrases, list):
        return labels

    for p in hate_phrases:
        if isinstance(p, dict):
            cp = p.get("char_pos", "")
            if isinstance(cp, str) and "-" in cp:
                s, e = parse_char_pos(cp)
                for i, (ts, te) in enumerate(offsets):
                    if not (te <= s or ts >= e):
                        labels[i] = 1
    return labels

def build_pred_token_labels(text: str, pred_spans: List[str]) -> List[int]:
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_len,
        return_offsets_mapping=True,
        add_special_tokens=True
    )
    offsets = enc["offset_mapping"]

    char_mask = [0] * len(text)
    lower_text = text.lower()

    if isinstance(pred_spans, list):
        for span in pred_spans:
            if isinstance(span, str):
                s = span.strip().lower()
                if not s:
                    continue
                start = 0
                while True:
                    idx = lower_text.find(s, start)
                    if idx == -1:
                        break
                    for c in range(idx, min(idx + len(s), len(char_mask))):
                        char_mask[c] = 1
                    start = idx + len(s)

    labels = [0] * len(offsets)
    for i, (ts, te) in enumerate(offsets):
        if ts < len(char_mask):
            if any(char_mask[ts: min(te, len(char_mask))]):
                labels[i] = 1
    return labels

# ====================
# Parse model output list
# ====================
def safe_parse_list(ans_text: str) -> List[str]:
    if not isinstance(ans_text, str):
        return []
    start = ans_text.find("[")
    end = ans_text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    sub = ans_text[start:end+1]
    try:
        obj = json.loads(sub)
        if isinstance(obj, list):
            return [s.strip() for s in obj if isinstance(s, str) and s.strip()]
    except:
        pass
    return []

# ====================
# Eval (loc only)
# ====================
@torch.no_grad()
def evaluate_t3_loc(model, dataset_raw, desc="dev"):
    model.eval()
    loc_gold_all, loc_pred_all = [], []
    parse_err = 0

    for i in range(len(dataset_raw)):
        sraw = dataset_raw[i]
        text = sraw.get("text", "")
        gold_hate = sraw.get("hate_phrases", [])

        prompt = (
            T3_INSTRUCTION + "\n\n"
            + f"Text: {text}\n"
            + "Answer:\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0
        )
        gen = gen_ids[0][inputs.input_ids.shape[1]:]
        out = tokenizer.decode(gen, skip_special_tokens=True).strip()

        pred_spans = safe_parse_list(out)
        if out.find("[") == -1 or out.find("]") == -1:
            parse_err += 1

        gold_tok = build_gold_token_labels(text, gold_hate)
        pred_tok = build_pred_token_labels(text, pred_spans)
        L = min(len(gold_tok), len(pred_tok))
        loc_gold_all.extend(gold_tok[:L])
        loc_pred_all.extend(pred_tok[:L])

    if len(loc_gold_all) > 0:
        _, _, loc_f1_macro, _ = precision_recall_fscore_support(
            loc_gold_all, loc_pred_all, average="macro", zero_division=0
        )
    else:
        loc_f1_macro = 0.0

    metrics = {
        f"{desc}_loc_f1_macro": loc_f1_macro,
        f"{desc}_parse_error_rate": (parse_err / max(1, len(dataset_raw))),
    }

    print(f"\n===== {desc} evaluation (T3) =====")
    print(f"{desc}_loc_f1_macro: {loc_f1_macro:.4f}")
    print(f"{desc}_parse_error_rate: {metrics[f'{desc}_parse_error_rate']:.4%}")
    print("=================================\n")
    return metrics

# ====================
# Callback: save best by dev_loc_f1_macro
# ====================
class T3EvalCallback(TrainerCallback):
    def __init__(self, dev_raw, out_dir):
        self.dev_raw = dev_raw
        self.out_dir = out_dir
        self.best = -1.0

    def on_epoch_end(self, args, state, control, **kwargs):
        m = kwargs["model"]
        metrics = evaluate_t3_loc(m, self.dev_raw, desc="dev")
        score = metrics["dev_loc_f1_macro"]
        if score > self.best:
            self.best = score
            best_dir = os.path.join(self.out_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            print(f"New best dev_loc_f1_macro={score:.4f}, saving LoRA to {best_dir}")
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
    callbacks=[T3EvalCallback(dev_raw, output_dir)]
)

# ====================
# Main
# ====================
if __name__ == "__main__":
    print("===== Start QLoRA SFT: T3 =====")
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

    print("===== Test evaluation (T3 loc) =====")
    test_metrics = evaluate_t3_loc(best_model_for_test, test_raw, desc="test")

    # save test predictions (phrases + raw output)
    results = []
    best_model_for_test.eval()
    for i in range(len(test_raw)):
        sraw = test_raw[i]
        text = sraw.get("text", "")
        gold_hate = sraw.get("hate_phrases", [])

        prompt = (
            T3_INSTRUCTION + "\n\n"
            + f"Text: {text}\n"
            + "Answer:\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(best_model_for_test.device)
        with torch.no_grad():
            gen_ids = best_model_for_test.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0
            )
        gen = gen_ids[0][inputs.input_ids.shape[1]:]
        out = tokenizer.decode(gen, skip_special_tokens=True).strip()
        pred_spans = safe_parse_list(out)

        gold_contents = []
        if isinstance(gold_hate, list):
            for p in gold_hate:
                if isinstance(p, dict) and isinstance(p.get("content", ""), str):
                    c = p["content"].strip()
                    if c:
                        gold_contents.append(c)

        results.append({
            "id": sraw.get("id", ""),
            "text": text,
            "gold_phrases": gold_contents,
            "pred_phrases": pred_spans,
            "raw_output": out
        })

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "test_predictions.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(os.path.join(output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    print("Saved predictions and metrics!")
