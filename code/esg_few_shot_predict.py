import json
import random
import re
import warnings
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

# ─────────────────────────── Step 1: 資料讀取 ───────────────────────────
DATA_PATH = "D:/my_task/2026/AICUP/data/vpesg4k_train_1000.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# ─────────────────────────── Step 2: 資料前處理 ─────────────────────────
DROP_COLS = {"esg_type", "company", "ticker", "page_number", "pdf_url", "company_source"}
KEEP_COLS = ["id", "data", "promise_status", "promise_string",
             "verification_timeline", "evidence_status",
             "evidence_string", "evidence_quality"]

data = [{k: v for k, v in item.items() if k not in DROP_COLS} for item in raw_data]

# ─────────────────────────── Step 3: 資料切割 ───────────────────────────
random.seed(42)

yes_items = [d for d in data if d["promise_status"] == "Yes"]
no_items  = [d for d in data if d["promise_status"] == "No"]

def split_8_2(items, seed=42):
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * 0.8)
    return shuffled[:cut], shuffled[cut:]

yes_train, yes_test = split_8_2(yes_items)
no_train,  no_test  = split_8_2(no_items)

train_set = yes_train + no_train
test_set  = yes_test  + no_test

random.Random(42).shuffle(train_set)
random.Random(42).shuffle(test_set)

print(f"Train size: {len(train_set)}, Test size: {len(test_set)}")

# ─────────────────────────── Step 4: 載入模型 ───────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "Qwen/Qwen3.5-4B"
print(f"Loading model: {MODEL_NAME} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print("Model loaded.")

# ─────────────────────────── Step 5: Few-shot Prompting ─────────────────

FIELD_DEFS = """\
Field definitions:
1. promise_status: Whether the text contains an ESG promise/commitment. Values: Yes | No
2. evidence_status: Evaluate whether commitment statements are semantically clear and unambiguous, and identify potential greenwashing risks.
Values: Yes | No | N/A
3. evidence_quality: Quality of the evidence provided.
   Values: Clear | Not Clear | Misleading | N/A
4. verification_timeline: When the promise is expected to be fulfilled.
   Values: already | within_2_years | between_2_and_5_years | longer_than_5_years | N/A
"""

def build_few_shot_prompt(examples, test_item):
    prompt_parts = [FIELD_DEFS, "\n--- Examples ---\n"]
    for ex in examples:
        prompt_parts.append(
            f"Text: {ex['data']}\n"
            f"promise_status: {ex['promise_status']}\n"
            f"verification_timeline: {ex['verification_timeline']}\n"
            f"evidence_status: {ex['evidence_status']}\n"
            f"evidence_quality: {ex['evidence_quality']}\n\n"
        )
    prompt_parts.append(
        f"--- Now classify the following text ---\n"
        f"Text: {test_item['data']}\n"
        "Answer in exactly this format (one value per line, no extra text):\n"
        "promise_status: <value>\n"
        "verification_timeline: <value>\n"
        "evidence_status: <value>\n"
        "evidence_quality: <value>\n"
    )
    return "".join(prompt_parts)


VALID_VALUES = {
    "promise_status":        {"Yes", "No"},
    "verification_timeline": {"already", "within_2_years", "between_2_and_5_years",
                               "longer_than_5_years", "N/A"},
    "evidence_status":       {"Yes", "No", "N/A"},
    "evidence_quality":      {"Clear", "Not Clear", "Misleading", "N/A"},
}

FIELD_DEFAULTS = {
    "promise_status":        "No",
    "verification_timeline": "N/A",
    "evidence_status":       "N/A",
    "evidence_quality":      "N/A",
}

def parse_output(text):
    results = {}
    for field in ["promise_status", "verification_timeline",
                  "evidence_status", "evidence_quality"]:
        # Try to extract the value after "field: "
        pattern = rf"{field}\s*:\s*(.+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1).strip().split()[0].strip(".,;")
            # Normalise case for N/A
            if val.upper() == "N/A":
                val = "N/A"
            if val in VALID_VALUES[field]:
                results[field] = val
            else:
                results[field] = FIELD_DEFAULTS[field]
        else:
            results[field] = FIELD_DEFAULTS[field]

    # 正規化：promise_status=No 時，其餘欄位強制為 N/A
    if results["promise_status"] == "No":
        results["verification_timeline"] = "N/A"
        results["evidence_status"]       = "N/A"
        results["evidence_quality"]      = "N/A"

    return results


def generate(prompt, max_new_tokens=128):
    messages = [
        {"role": "system", "content": "You are a ESG expert."},
        {"role": "user",   "content": prompt},
    ]
    # Use chat template if available, else fall back to plain prompt
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,   # Qwen3 non-thinking mode
        )
    except Exception:
        text = f"System: You are a ESG expert.\nUser: {prompt}\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


FEW_SHOT_K = 3
rng = random.Random(42)

FIELDS = ["promise_status", "verification_timeline", "evidence_status", "evidence_quality"]

predictions = {f: [] for f in FIELDS}
ground_truth = {f: [] for f in FIELDS}
result_rows  = []   # 用於輸出 txt

print(f"\nRunning predictions on {len(test_set)} test items...")
for idx, item in enumerate(test_set):
    examples = rng.sample(train_set, FEW_SHOT_K)
    prompt   = build_few_shot_prompt(examples, item)
    output   = generate(prompt)
    preds    = parse_output(output)

    for field in FIELDS:
        predictions[field].append(preds[field])
        ground_truth[field].append(item[field])

    result_rows.append({
        "id":                    item["id"],
        "promise_status_pred":   preds["promise_status"],
        "promise_status_true":   item["promise_status"],
        "verification_timeline_pred": preds["verification_timeline"],
        "verification_timeline_true": item["verification_timeline"],
        "evidence_status_pred":  preds["evidence_status"],
        "evidence_status_true":  item["evidence_status"],
        "evidence_quality_pred": preds["evidence_quality"],
        "evidence_quality_true": item["evidence_quality"],
        "raw_output":            output.strip(),
    })

    if (idx + 1) % 10 == 0:
        print(f"  [{idx+1}/{len(test_set)}] done")

# ─────────────────────────── Step 6: 寫出結果 txt ───────────────────────
import os
from datetime import datetime

OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"predictions_{timestamp}.txt")

with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    header = (
        f"{'ID':<8} "
        f"{'ps_pred':<12} {'ps_true':<12} "
        f"{'vt_pred':<26} {'vt_true':<26} "
        f"{'es_pred':<10} {'es_true':<10} "
        f"{'eq_pred':<12} {'eq_true':<12} "
        f"raw_output\n"
    )
    fout.write(header)
    fout.write("-" * 160 + "\n")
    for row in result_rows:
        line = (
            f"{row['id']:<8} "
            f"{row['promise_status_pred']:<12} {row['promise_status_true']:<12} "
            f"{row['verification_timeline_pred']:<26} {row['verification_timeline_true']:<26} "
            f"{row['evidence_status_pred']:<10} {row['evidence_status_true']:<10} "
            f"{row['evidence_quality_pred']:<12} {row['evidence_quality_true']:<12} "
            f"{repr(row['raw_output'])}\n"
        )
        fout.write(line)

print(f"\nPrediction results saved to: {OUTPUT_PATH}")

# ─────────────────────────── Step 7: 評分 ───────────────────────────────
WEIGHTS = {
    "promise_status":        0.20,
    "verification_timeline": 0.15,
    "evidence_status":       0.30,
    "evidence_quality":      0.35,
}

print("\n===== Evaluation Results =====")
score_lines = []
weighted_sum = 0.0
for field, weight in WEIGHTS.items():
    labels = sorted(VALID_VALUES[field])
    score = f1_score(
        ground_truth[field],
        predictions[field],
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_sum += weight * score
    line = f"  {field:<26} Macro F1 = {score:.4f}  (weight={weight})"
    score_lines.append(line)
    print(line)

final_line = f"\n  Weighted Macro F1 Score = {weighted_sum:.4f}"
print(final_line)

# 將評分結果追加寫入同一個 txt
with open(OUTPUT_PATH, "a", encoding="utf-8") as fout:
    fout.write("\n" + "=" * 50 + "\n")
    fout.write("Evaluation Results\n")
    fout.write("=" * 50 + "\n")
    for line in score_lines:
        fout.write(line.strip() + "\n")
    fout.write(final_line.strip() + "\n")

print(f"Scores appended to: {OUTPUT_PATH}")
