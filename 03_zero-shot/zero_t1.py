import os
import re
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# =========================
# 1) Model / Tokenizer
# =========================

# model_path = "/media/qust521/92afc6a9-13fd-4458-8a46-4b008127de08/LLMs/Qwen2-7B-Instruct"
model_path="/media/qust521/92afc6a9-13fd-4458-8a46-4b008127de08/LLMs/Phi-3-mini-128k-instruct"


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    # trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# =========================
# ) Run + Save (输出你要的格式)
# =========================
input_jsonl = "/home/qust521/Projects/ljy/Complementary experiment/data.jsonl"

model_name = os.path.basename(model_path.rstrip("/"))
model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)  # 防止奇怪字符导致路径不合法
output_dir = "./results"
os.makedirs(output_dir, exist_ok=True)
output_jsonl = os.path.join(output_dir, f"{model_name}_result_T1.jsonl")
# =========================
# 2) Target profiles
# =========================
target_profiles = {
}
# =========================
# 3) Labels + Prompt (对齐)
# =========================
# T1 数字标签映射：targeted-abusive:1; unidentified-targets:0; non-abusive:2
TRI_LABEL2ID = {
    "targeted-abusive": "1",
    "unidentified-targets": "0",
    "non-abusive": "2",
}
ID_LABELS = ["0", "1", "2"]

SYSTEM_PROMPT = """You are a content review expert. Given a text and a specific @Twitter account (the target), determine whether the text contains hate speech/abusive language and whether it is directed at that target.

You will also be given a brief profile of the target account. Use the target profile ONLY to:
- resolve coreference (e.g., "she/he/they", "your father", role/title),
- disambiguate which person/account is being referred to,
- understand context about the target (public role, identity, etc.).
Do NOT use the profile to assume abuse when the text itself is not abusive.

Choose exactly ONE of the following three classes, and output ONLY its ID:
- 0 = unidentified-targets (hate speech/abusive language present but NOT directed at the given target)
- 1 = targeted-abusive (hate speech/abusive language directed at the given target)
- 2 = non-abusive (no hate speech/abusive language)

Important:
- Even if multiple accounts are mentioned, judge targeting ONLY with respect to the given target account.
- Output ONLY one character: 0 or 1 or 2. No explanation.
"""

# =========================
# 4) IO Helpers
# =========================
def read_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def build_user_content(text: str, target: str) -> str:
    tgt = (target or "").strip()
    profile = target_profiles.get(tgt, "No profile available.")
    return (
        f"Target account: {tgt}\n"
        f"Target profile: {profile}\n\n"
        f"Text:\n\"{text}\""
    )

def normalize_label(raw_text: str) -> str:
    """
    把模型输出归一成三类之一（只返回 0/1/2 字符串）
    """
    s = (raw_text or "").strip()
    s_low = s.lower()

    # 先做精确匹配/包含匹配
    for lab in ID_LABELS:
        if s_low == lab:
            return lab
    for lab in ID_LABELS:
        if lab in s_low:
            return lab

    return "PARSE_ERROR"

# =========================
# 5) Constrained decoding (强制只生成三类ID)
# =========================
eos_id = tokenizer.eos_token_id
if eos_id is None:
    eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

# 允许的输出序列（token级）
allowed_seqs = []
for lab in ID_LABELS:
    for variant in [lab, " " + lab, "\n" + lab]:
        tok_seq = tokenizer.encode(variant, add_special_tokens=False)
        allowed_seqs.append((tok_seq, lab))

max_seq_len = max(len(x[0]) for x in allowed_seqs)

def make_prefix_allowed_tokens_fn(prompt_len: int):
    def prefix_allowed_tokens_fn(batch_id, input_ids):
        gen_ids = input_ids[prompt_len:].detach().cpu().tolist()
        allowed_next = set()

        for seq, _lab in allowed_seqs:
            if len(gen_ids) <= len(seq) and gen_ids == seq[:len(gen_ids)]:
                if len(gen_ids) == len(seq):
                    allowed_next.add(eos_id)
                else:
                    allowed_next.add(seq[len(gen_ids)])

        return list(allowed_next) if allowed_next else [eos_id]
    return prefix_allowed_tokens_fn

def generate_label(text: str, target: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(text, target)}
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)
    prompt_len = model_inputs.input_ids.shape[1]

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_seq_len + 1,
        do_sample=False,
        temperature=0.0,
        num_beams=1,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
        prefix_allowed_tokens_fn=make_prefix_allowed_tokens_fn(prompt_len)
    )

    gen = generated_ids[0][prompt_len:]
    raw = tokenizer.decode(gen, skip_special_tokens=True)
    return normalize_label(raw)

data = read_jsonl(input_jsonl)

# ===== Eval buffers (T1) =====
y_true = []
y_pred = []
parse_err = 0
total = 0

with open(output_jsonl, "w", encoding="utf-8") as f_out:
    for item in tqdm(data, desc="Zero-shot tri-class", unit="item"):
        text = item.get("text", "")
        target = item.get("target", "")
        true_label = item.get("tri_label", "")

        true_id = TRI_LABEL2ID.get(true_label, "PARSE_ERROR")
        pred_id = generate_label(text, target)

        # ===== collect for evaluation =====
        total += 1
        y_true.append(true_id)
        y_pred.append(pred_id)
        if pred_id == "PARSE_ERROR":
            parse_err += 1

        f_out.write(json.dumps({"true": true_id, "pred": pred_id}, ensure_ascii=False) + "\n")

print(f"Done. Saved to: {output_jsonl}")

# ===== Evaluation (T1) =====
tri_acc = accuracy_score(y_true, y_pred)
_, _, tri_f1_macro, _ = precision_recall_fscore_support(
    y_true, y_pred,
    labels=ID_LABELS,
    average="macro",
    zero_division=0
)

print("\n===== T1 Evaluation (tri-class) =====")
print(f"tri_accuracy: {tri_acc:.4f}")
print(f"tri_f1_macro: {tri_f1_macro:.4f}")
if total > 0:
    print(f"parse_error_rate: {parse_err}/{total} = {parse_err/total:.4%}")
else:
    print("parse_error_rate: 0/0 = 0.0000%")
print("=====================================\n")

metrics_txt = os.path.join(output_dir, f"{model_name}_T1_metrics.txt")

lines = []
lines.append("\n===== T1 Evaluation (tri-class) =====")
lines.append(f"tri_accuracy: {tri_acc:.4f}")
lines.append(f"tri_f1_macro: {tri_f1_macro:.4f}")
if total > 0:
    lines.append(f"parse_error_rate: {parse_err}/{total} = {parse_err/total:.4%}")
else:
    lines.append("parse_error_rate: 0/0 = 0.0000%")
lines.append("=====================================\n")

# 打印到屏幕
for line in lines:
    print(line)

# 保存到txt
with open(metrics_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(lines).strip() + "\n")

print(f"Saved metrics to: {metrics_txt}")


