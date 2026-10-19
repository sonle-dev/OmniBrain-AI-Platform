import os
import json
import random
import re
from typing import List, Dict, Any

# Fix random seed for reproducible benchmark splits
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

BASE_DIR = r"D:\OmniBrain AI Platform"
GRAPHRAG_DIR = os.path.join(BASE_DIR, "graphrag_data")
BENCHMARK_DIR = os.path.join(BASE_DIR, "benchmark_data")
HOLDOUT_TEST_DIR = os.path.join(BENCHMARK_DIR, "test_holdout")

os.makedirs(BENCHMARK_DIR, exist_ok=True)
os.makedirs(HOLDOUT_TEST_DIR, exist_ok=True)

# Load master knowledge store
master_store_path = os.path.join(GRAPHRAG_DIR, "lms", "knowledge_store.json")
if not os.path.exists(master_store_path):
    raise FileNotFoundError(f"Master knowledge store not found at {master_store_path}")

with open(master_store_path, "r", encoding="utf-8") as f:
    all_chunks: List[Dict[str, Any]] = json.load(f)

print(f"Loaded total {len(all_chunks)} chunks from master store.")

# Group chunks by courseId and session to ensure balanced stratified splitting
by_course_session: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
for c in all_chunks:
    cid = c.get("courseId", "unknown")
    sess = c.get("session", 1)
    if cid not in by_course_session:
        by_course_session[cid] = {}
    if sess not in by_course_session[cid]:
        by_course_session[cid][sess] = []
    by_course_session[cid][sess].append(c)

train_chunks = []
val_chunks = []
test_chunks = []

# Stratified split: 70% Train, 15% Validation, 15% Test within each course and session
for cid, sessions in sorted(by_course_session.items()):
    for sess, chunks in sorted(sessions.items()):
        # Deterministic shuffle for each session
        shuffled = list(chunks)
        random.shuffle(shuffled)
        
        n = len(shuffled)
        n_test = max(1, round(n * 0.15))
        n_val = max(1, round(n * 0.15))
        n_train = n - n_val - n_test
        
        # In case n_train <= 0 for tiny lists
        if n_train <= 0:
            n_train = max(1, n - 2)
            n_val = 1
            n_test = n - n_train - n_val

        s_train = shuffled[:n_train]
        s_val = shuffled[n_train:n_train + n_val]
        s_test = shuffled[n_train + n_val:]

        train_chunks.extend(s_train)
        val_chunks.extend(s_val)
        test_chunks.extend(s_test)

print(f"--- PHÂN CHIA 3 TẬP DỮ LIỆU CHUẨN ---")
print(f"1. Tập Huấn Luyện (Train Set):   {len(train_chunks)} chunks ({len(train_chunks)/len(all_chunks)*100:.1f}%)")
print(f"2. Tập Xác Thực (Validation Set): {len(val_chunks)} chunks ({len(val_chunks)/len(all_chunks)*100:.1f}%)")
print(f"3. Tập Kiểm Thử (Holdout Test):  {len(test_chunks)} chunks ({len(test_chunks)/len(all_chunks)*100:.1f}%)")

# Save datasets
train_file = os.path.join(BENCHMARK_DIR, "train_dataset.json")
val_file = os.path.join(BENCHMARK_DIR, "val_dataset.json")
test_file = os.path.join(HOLDOUT_TEST_DIR, "test_dataset_holdout.json")

with open(train_file, "w", encoding="utf-8") as f:
    json.dump(train_chunks, f, ensure_ascii=False, indent=2)

with open(val_file, "w", encoding="utf-8") as f:
    json.dump(val_chunks, f, ensure_ascii=False, indent=2)

with open(test_file, "w", encoding="utf-8") as f:
    json.dump(test_chunks, f, ensure_ascii=False, indent=2)

print(f"Saved Train set to: {train_file}")
print(f"Saved Val set to:   {val_file}")
print(f"Saved Test set to:  {test_file} (ISOLATED & SEALED)")

# Generate Test QA Benchmark from Holdout Test Chunks
test_qa_list = []
for idx, chunk in enumerate(test_chunks):
    text = chunk.get("text", "")
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 30 and not l.strip().startswith("---")]
    keywords = chunk.get("keywords", [])
    
    kw_str = ", ".join(keywords[:3]) if keywords else "nội dung kiến thức"
    course_name = chunk.get("courseTitle", "")
    sess_num = chunk.get("session", 1)
    
    question = f"Trong môn '{course_name}' (Buổi {sess_num}), hãy giải thích nguyên lý và đặc điểm chính liên quan đến {kw_str}."
    
    qa_item = {
        "id": f"TEST-QA-{idx+1:04d}",
        "courseId": chunk.get("courseId"),
        "courseTitle": course_name,
        "session": sess_num,
        "chunkId": chunk.get("id"),
        "page": chunk.get("page"),
        "keywords": keywords,
        "question": question,
        "ground_truth_context": text[:800],
        "sealed": True
    }
    test_qa_list.append(qa_item)

test_qa_file = os.path.join(HOLDOUT_TEST_DIR, "test_qa_benchmark.json")
with open(test_qa_file, "w", encoding="utf-8") as f:
    json.dump(test_qa_list, f, ensure_ascii=False, indent=2)

print(f"Generated {len(test_qa_list)} benchmark evaluation questions in: {test_qa_file}")
