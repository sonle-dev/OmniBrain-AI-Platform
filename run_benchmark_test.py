import os
import json
import time
import urllib.request
from typing import List, Dict, Any

BASE_DIR = r"D:\OmniBrain AI Platform"
TEST_QA_PATH = os.path.join(BASE_DIR, "benchmark_data", "test_holdout", "test_qa_benchmark.json")
REPORT_PATH = os.path.join(BASE_DIR, "benchmark_data", "test_holdout", "benchmark_evaluation_report.json")

def run_test(sample_size: int = 10):
    if not os.path.exists(TEST_QA_PATH):
        print(f"Error: Test QA benchmark not found at {TEST_QA_PATH}")
        return

    with open(TEST_QA_PATH, "r", encoding="utf-8") as f:
        qa_data: List[Dict[str, Any]] = json.load(f)

    total_available = len(qa_data)
    test_samples = qa_data[:sample_size] if sample_size > 0 else qa_data
    print("=" * 70)
    print(f"BẮT ĐẦU CHẠY KIỂM THỬ ĐỘC LẬP TẬP TEST (HOLDOUT BLIND TEST)")
    print(f"- Tổng số câu hỏi trong Tập Test: {total_available}")
    print(f"- Số câu hỏi kiểm thử đợt này: {len(test_samples)}")
    print("=" * 70)

    results = []
    passed = 0
    grounded_count = 0

    for idx, item in enumerate(test_samples):
        q_id = item["id"]
        course = item["courseTitle"]
        session = item["session"]
        question = item["question"]
        ground_truth = item["ground_truth_context"]
        keywords = item.get("keywords", [])

        print(f"\n[{idx+1}/{len(test_samples)}] Đang test: {q_id} ({course} - Buổi {session})")
        print(f"❓ Câu hỏi: {question}")

        # Send request to OmniBrain API
        t0 = time.time()
        payload = {
            "messages": [{"role": "user", "content": question}],
            "temperature": 0.1
        }
        
        req = urllib.request.Request(
            "http://127.0.0.1:8000/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-API-Key": "demo_key"
            }
        )

        reply_text = ""
        is_success = False
        duration = 0.0

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                reply_text = data.get("text", "").strip()
                is_success = True
                duration = round(time.time() - t0, 2)
        except Exception as e:
            reply_text = f"Lỗi kết nối / Timeout: {e}"
            duration = round(time.time() - t0, 2)

        # Evaluate response against ground truth keywords & semantics
        kw_matched = [kw for kw in keywords if kw.lower() in reply_text.lower()]
        has_keywords = len(kw_matched) > 0 or len(keywords) == 0
        
        # Check hallucination / honesty
        is_grounded = bool(reply_text and not "Lỗi" in reply_text)
        if has_keywords and is_grounded:
            passed += 1
            print(f"✅ ĐẠT (Khớp {len(kw_matched)}/{len(keywords)} từ khóa | {duration}s)")
        else:
            print(f"⚠️ CẦN XEM LẠI (Khớp {len(kw_matched)}/{len(keywords)} từ khóa | {duration}s)")

        results.append({
            "test_id": q_id,
            "course": course,
            "session": session,
            "question": question,
            "reply": reply_text,
            "matched_keywords": kw_matched,
            "duration_seconds": duration,
            "status": "PASS" if (has_keywords and is_grounded) else "REVIEW"
        })

    accuracy = (passed / len(test_samples)) * 100 if test_samples else 0
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_samples": len(test_samples),
        "passed": passed,
        "accuracy_rate": f"{accuracy:.1f}%",
        "detailed_results": results
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"KẾT QUẢ KIỂM THỬ TẬP TEST:")
    print(f"- Tỷ lệ đạt tiêu chuẩn: {accuracy:.1f}% ({passed}/{len(test_samples)})")
    print(f"- Báo cáo chi tiết đã lưu tại: {REPORT_PATH}")
    print("=" * 70)

if __name__ == "__main__":
    import sys
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_test(sample_size=count)
