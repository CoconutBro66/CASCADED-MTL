import os
import re
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support

# =========================
# 0) Paths
# =========================
model_path = "/media/qust521/92afc6a9-13fd-4458-8a46-4b008127de08/LLMs/Qwen2-7B-Instruct"
# model_path="/media/qust521/92afc6a9-13fd-4458-8a46-4b008127de08/LLMs/Phi-3-mini-128k-instruct"

input_jsonl = "/home/qust521/Projects/ljy/Complementary experiment/data.jsonl"

model_name = os.path.basename(model_path.rstrip("/"))
model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)  # 防止奇怪字符导致路径不合法
output_dir = "./results"
os.makedirs(output_dir, exist_ok=True)
output_jsonl = os.path.join(output_dir, f"{model_name}_result_T3.jsonl")

# offsets 需要的最大长度（不要太小，否则 char_pos 可能落在截断后文本外）
max_seq_len = 512

# =========================
# 1) Load model/tokenizer
# =========================
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    # trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# =========================
# 2) T3 Prompt: extract hate phrases only
# =========================
T3_SYSTEM_PROMPT = """You are a content review expert.

Task:
Extract all hateful/abusive phrases from the given text.

Output format:
Return ONLY a JSON array (list) of strings, e.g.:
["phrase1", "phrase2"]

Rules:
- Each phrase MUST appear verbatim in the input text (copy exact substrings).
- If there is no hateful phrase, output [].
- Output ONLY the JSON array. No explanation, no extra text.
"""

# =========================
# 3) IO
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
# 4) Gold char_pos -> token labels
# =========================
def parse_char_pos(char_pos: str):
    try:
        s, e = char_pos.split("-")
        s, e = int(s), int(e)
        if e < s:
            e = s
        return s, e
    except:
        return 0, 0

def build_gold_token_labels(text: str, hate_phrases):
    """
    hate_phrases: list[dict] with keys content/char_pos
    """
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
                    # offsets 对 special tokens 可能是 (0,0)，不影响：不会与正常区间重叠
                    if not (te <= s or ts >= e):
                        labels[i] = 1

    return labels

# =========================
# 5) Pred phrases -> token labels (by string matching)
# =========================
def build_pred_token_labels(text: str, pred_spans):
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

# =========================
# 6) Parse model output (strict; no guessing)
# =========================
def safe_parse_json_list(ans_text: str):
    """
    只接受 JSON list；解析失败返回 "PARSE_ERROR"
    """
    if not isinstance(ans_text, str):
        return "PARSE_ERROR", []

    start = ans_text.find("[")
    end = ans_text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return "PARSE_ERROR", []

    sub = ans_text[start:end+1]
    try:
        obj = json.loads(sub)
        if not isinstance(obj, list):
            return "PARSE_ERROR", []
        spans = [s.strip() for s in obj if isinstance(s, str) and s.strip()]
        return "OK", spans
    except:
        return "PARSE_ERROR", []

# =========================
# 7) Zero-shot generate hate phrases
# =========================
def generate_hate_phrases(text: str):
    messages = [
        {"role": "system", "content": T3_SYSTEM_PROMPT},
        {"role": "user", "content": f'Text:\n"{text}"'}
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        gen_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0
        )

    gen = gen_ids[0][model_inputs.input_ids.shape[1]:]
    raw = tokenizer.decode(gen, skip_special_tokens=True).strip()
    status, spans = safe_parse_json_list(raw)
    return status, spans, raw

# =========================
# 8) Run: only loc evaluation
# =========================
data = read_jsonl(input_jsonl)

loc_gold_all = []
loc_pred_all = []
parse_error_cnt = 0
used_cnt = 0

with open(output_jsonl, "w", encoding="utf-8") as f_out:
    for item in tqdm(data, desc="Zero-shot T3 (loc only)", unit="item"):
        # 只对 targeted-abusive 做定位评估（和你微调代码一致）
        if item.get("tri_label", "") != "targeted-abusive":
            continue

        text = item.get("text", "")
        gold_hate = item.get("hate_phrases", [])

        status, pred_spans, raw = generate_hate_phrases(text)
        if status != "OK":
            parse_error_cnt += 1

        gold_tok = build_gold_token_labels(text, gold_hate)
        pred_tok = build_pred_token_labels(text, pred_spans)

        L = min(len(gold_tok), len(pred_tok))
        loc_gold_all.extend(gold_tok[:L])
        loc_pred_all.extend(pred_tok[:L])
        used_cnt += 1

        # 保存每条样本的预测（便于排查；不影响评估“只算loc”）
        gold_contents = []
        if isinstance(gold_hate, list):
            for p in gold_hate:
                if isinstance(p, dict) and isinstance(p.get("content", ""), str):
                    c = p["content"].strip()
                    if c:
                        gold_contents.append(c)

        out_line = {
            "id": item.get("id", ""),
            "parse_status": status,     # OK / PARSE_ERROR
            "gold_phrases": gold_contents,
            "pred_phrases": pred_spans,
            "raw_output": raw
        }
        f_out.write(json.dumps(out_line, ensure_ascii=False) + "\n")

# loc token-level F1 (macro over {0,1})
if len(loc_gold_all) > 0:
    _, _, loc_f1_macro, _ = precision_recall_fscore_support(
        loc_gold_all, loc_pred_all, average="macro", zero_division=0
    )
else:
    loc_f1_macro = 0.0

print("\n===== T3 Localization (token-level) =====")
print(f"Evaluated samples (tri_label=targeted-abusive): {used_cnt}")
print(f"Parse errors: {parse_error_cnt}")
print(f"loc_f1_macro: {loc_f1_macro:.4f}")
print(f"Predictions saved to: {output_jsonl}")
print("========================================\n")

metrics_txt = os.path.join(output_dir, f"{model_name}_T3_metrics.txt")

lines = []
lines.append("\n===== T3 Localization (token-level) =====")
lines.append(f"Evaluated samples (tri_label=targeted-abusive): {used_cnt}")
lines.append(f"Parse errors: {parse_error_cnt}")
lines.append(f"loc_f1_macro: {loc_f1_macro:.4f}")
lines.append(f"Predictions saved to: {output_jsonl}")
lines.append("========================================\n")

# 打印到屏幕
for line in lines:
    print(line)

# 保存到txt
with open(metrics_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(lines).strip() + "\n")

print(f"Saved metrics to: {metrics_txt}")

