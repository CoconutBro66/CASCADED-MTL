import os
import re
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# =========================
# 0) 路径配置：只需要改 input_jsonl
# =========================
# model_path = "/media/qust521/92afc6a9-13fd-4458-8a46-4b008127de08/LLMs/Qwen2-7B-Instruct"
model_path="/media/qust521/92afc6a9-13fd-4458-8a46-4b008127de08/LLMs/Phi-3-mini-128k-instruct"

input_jsonl = "/home/qust521/Projects/ljy/Complementary experiment/data.jsonl"

model_name = os.path.basename(model_path.rstrip("/"))
model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
output_dir = "./results"
os.makedirs(output_dir, exist_ok=True)
output_jsonl = os.path.join(output_dir, f"{model_name}_result_T2.jsonl")

# =========================
# 1) 加载模型
# =========================
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    # trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

eos_id = tokenizer.eos_token_id
if eos_id is None:
    eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

# =========================
# 2) T2 标签集合 + Prompt（输出为ID）
# =========================
VALID_FINE_LABELS = [
    "death_threat", "sexual_assault", "sexual_explicit", "physical_harm",
    "radiation_of_threats", "attacks_on_credibility", "misogynistic",
    "homophobic", "religious", "political_sectarian", "racist", "general"
]

FINE_LABEL2ID = {lab: str(i) for i, lab in enumerate(VALID_FINE_LABELS)}
ID_FINE_LABELS = [str(i) for i in range(len(VALID_FINE_LABELS))]

# 在 prompt 里显示 id->label 映射
FINE_ID_TEXT = "\n".join([f"{i} = {lab}" for i, lab in enumerate(VALID_FINE_LABELS)])

FINE_SYSTEM_PROMPT = f"""You are a classification expert.

The input text is already confirmed to be targeted-abusive.
Your job is to choose exactly ONE fine-grained hate/abuse category ID.

Choose exactly ONE ID from this closed set:
{FINE_ID_TEXT}

Output ONLY the ID (an integer from 0 to 11). No explanation, no extra text.
"""

# =========================
# 3) JSONL 读取
# =========================
def read_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

# =========================
# 4) 受限解码：强制只能输出 0..11
# =========================
def make_allowed_seqs(labels):
    allowed = []
    for lab in labels:
        for variant in [lab, " " + lab, "\n" + lab]:
            tok = tokenizer.encode(variant, add_special_tokens=False)
            allowed.append((tok, lab))
    return allowed

FINE_ALLOWED = make_allowed_seqs(ID_FINE_LABELS)
max_fine_len = max(len(seq) for seq, _ in FINE_ALLOWED)

def make_prefix_allowed_tokens_fn(prompt_len: int, allowed_seqs):
    def _fn(batch_id, input_ids):
        gen_ids = input_ids[prompt_len:].detach().cpu().tolist()
        allowed_next = set()

        for seq, _canon in allowed_seqs:
            if len(gen_ids) <= len(seq) and gen_ids == seq[:len(gen_ids)]:
                if len(gen_ids) == len(seq):
                    allowed_next.add(eos_id)
                else:
                    allowed_next.add(seq[len(gen_ids)])

        return list(allowed_next) if allowed_next else [eos_id]
    return _fn

def generate_fine_label(text: str) -> str:
    messages = [
        {"role": "system", "content": FINE_SYSTEM_PROMPT},
        {"role": "user", "content": f'Text:\n"{text}"'}
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)
    prompt_len = model_inputs.input_ids.shape[1]

    gen_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_fine_len + 1,
        do_sample=False,
        temperature=0.0,
        num_beams=1,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
        prefix_allowed_tokens_fn=make_prefix_allowed_tokens_fn(prompt_len, FINE_ALLOWED),
    )

    gen = gen_ids[0][prompt_len:]
    raw = tokenizer.decode(gen, skip_special_tokens=True).strip()

    raw_low = raw.lower()
    if raw_low in ID_FINE_LABELS:
        return raw_low

    return "PARSE_ERROR"

# =========================
# 5) 主程序：只对 gold tri_label=targeted-abusive 跑 T2
# =========================
data = read_jsonl(input_jsonl)

written = 0

# ===== Eval buffers (T2) =====
y_true = []
y_pred = []
parse_err = 0
total = 0

with open(output_jsonl, "w", encoding="utf-8") as f_out:
    for item in tqdm(data, desc="Zero-shot T2 (12-way)", unit="item"):
        if item.get("tri_label", "") != "targeted-abusive":
            continue

        text = item.get("text", "")
        true_fine = item.get("fine_label", None)

        true_id = FINE_LABEL2ID.get(true_fine, "PARSE_ERROR")
        pred_id = generate_fine_label(text)

        # ===== collect for evaluation =====
        total += 1
        y_true.append(true_id)
        y_pred.append(pred_id)
        if pred_id == "PARSE_ERROR":
            parse_err += 1

        out_line = {"true": true_id, "pred": pred_id}
        f_out.write(json.dumps(out_line, ensure_ascii=False) + "\n")
        written += 1

print(f"Done. Wrote {written} lines to: {output_jsonl}")

# ===== Evaluation (T2) =====
fine_acc = accuracy_score(y_true, y_pred)
_, _, fine_f1_macro, _ = precision_recall_fscore_support(
    y_true, y_pred,
    labels=ID_FINE_LABELS,
    average="macro",
    zero_division=0
)

print("\n===== T2 Evaluation (12-way fine) =====")
# print(f"fine_accuracy: {fine_acc:.4f}")
print(f"fine_f1_macro: {fine_f1_macro:.4f}")
if total > 0:
    print(f"parse_error_rate: {parse_err}/{total} = {parse_err/total:.4%}")
else:
    print("parse_error_rate: 0/0 = 0.0000%")
print("======================================\n")

metrics_txt = os.path.join(output_dir, f"{model_name}_T2_metrics.txt")

lines = []
lines.append("\n===== T2 Evaluation (12-way fine) =====")
# lines.append(f"fine_accuracy: {fine_acc:.4f}")  # 你已注释掉accuracy，这里也保持注释
lines.append(f"fine_f1_macro: {fine_f1_macro:.4f}")
if total > 0:
    lines.append(f"parse_error_rate: {parse_err}/{total} = {parse_err/total:.4%}")
else:
    lines.append("parse_error_rate: 0/0 = 0.0000%")
lines.append("======================================\n")

# 打印到屏幕
for line in lines:
    print(line)

# 保存到txt
with open(metrics_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(lines).strip() + "\n")

print(f"Saved metrics to: {metrics_txt}")

