import os
import json
import re
import networkx as nx
from typing import List, Dict, Any

BASE_DIR = r"D:\OmniBrain AI Platform"
GRAPHRAG_DIR = os.path.join(BASE_DIR, "graphrag_data")
BENCHMARK_DIR = os.path.join(BASE_DIR, "benchmark_data")

# 1. Load Train and Validation sets (Total known knowledge to be ingested)
train_file = os.path.join(BENCHMARK_DIR, "train_dataset.json")
val_file = os.path.join(BENCHMARK_DIR, "val_dataset.json")

with open(train_file, "r", encoding="utf-8") as f:
    train_chunks = json.load(f)
with open(val_file, "r", encoding="utf-8") as f:
    val_chunks = json.load(f)

# Known knowledge = Train (568) + Validation (123) = 691 chunks
# Test set (123 chunks) is COMPLETELY EXCLUDED!
known_chunks: List[Dict[str, Any]] = train_chunks + val_chunks
print(f"Total chunks to ingest into Knowledge Base (Train + Val): {len(known_chunks)}")
print(f"Holdout Test chunks EXCLUDED: 123 chunks (Sealed in benchmark_data/test_holdout)")

# Metadata map
COURSE_META = {
    "c-101": {"title": "Lập trình Enterprise với Java 17/21 & Spring Boot 3", "code": "CNTT.K22A.D1.K1.N01"},
    "c-102": {"title": "Kiến trúc & Tối ưu Cơ sở Dữ liệu PostgreSQL Enterprise", "code": "CNTT.K22B.D1.K2.N02"},
    "c-103": {"title": "Phương pháp phát triển phần mềm hướng đối tượng-2-25", "code": "CNTT.K22C.D1.K3.N03"},
    "c-104": {"title": "Quản lý dự án CNTT-2-25", "code": "CNTT.K22D.D1.K4.N04"},
    "c-105": {"title": "Quản trị hệ thống-2-25", "code": "CNTT.K22E.D1.K5.N05"},
}

# Group chunks by course
chunks_by_course: Dict[str, List[Dict[str, Any]]] = {}
for c in known_chunks:
    cid = c.get("courseId", "unknown")
    if cid not in chunks_by_course:
        chunks_by_course[cid] = []
    chunks_by_course[cid].append(c)

global_graph = nx.MultiDiGraph()
global_store = []

# Cross-Course Bridge Definitions
CROSS_BRIDGES = [
    ("c-101", "c-102", "Spring Data JPA / HikariCP", "kết_nối_và_quản_trị", "PostgreSQL Enterprise"),
    ("c-103", "c-101", "Clean Architecture / SOLID", "định_hình_cấu_trúc_mã_nguồn", "Spring Boot Service Layer"),
    ("c-105", "c-101", "Docker Container / Linux Server", "môi_trường_triển_khai", "Spring Boot Application"),
    ("c-105", "c-102", "Linux Kernel & Systemd", "tối_ưu_hạ_tầng_cho", "PostgreSQL Database Cluster"),
    ("c-104", "c-103", "Agile / Scrum Sprint", "quản_trị_tiến_độ_phát_triển", "Object-Oriented Software"),
]

for cid, meta in COURSE_META.items():
    c_title = meta["title"]
    c_code = meta["code"]
    c_chunks = chunks_by_course.get(cid, [])
    
    out_dir = os.path.join(GRAPHRAG_DIR, cid)
    os.makedirs(out_dir, exist_ok=True)
    
    course_graph = nx.MultiDiGraph()
    course_graph.add_node(c_title, type="Course", id=cid, code=c_code)
    global_graph.add_node(c_title, type="Course", id=cid, code=c_code)
    
    # 1. Pipeline: Session 1 -> Session 2 -> ... -> Session 9
    for s_idx in range(1, 10):
        curr_session_id = f"{c_title} - Buổi {s_idx}"
        course_graph.add_node(curr_session_id, type="Session", number=s_idx, course=c_title)
        course_graph.add_edge(c_title, curr_session_id, relation="chương_trình_học")
        
        global_graph.add_node(curr_session_id, type="Session", number=s_idx, course=c_title)
        global_graph.add_edge(c_title, curr_session_id, relation="chương_trình_học")
        
        if s_idx > 1:
            prev_session_id = f"{c_title} - Buổi {s_idx - 1}"
            course_graph.add_edge(prev_session_id, curr_session_id, relation="tiền_đề_cho")
            global_graph.add_edge(prev_session_id, curr_session_id, relation="tiền_đề_cho")

    # 2. Add Chunks & Concept-to-Concept Mesh links
    for chunk in c_chunks:
        sess_num = chunk.get("session", 1)
        session_node_id = f"{c_title} - Buổi {sess_num}"
        keywords = chunk.get("keywords", [])
        
        # Link concepts to Session
        concepts = []
        for kw in keywords:
            term_norm = kw.strip().title() if len(kw) > 4 else kw.strip().upper()
            concepts.append(term_norm)
            course_graph.add_node(term_norm, type="Concept", course=c_title)
            course_graph.add_edge(session_node_id, term_norm, relation="giảng_dạy")
            
            global_graph.add_node(term_norm, type="Concept", course=c_title)
            global_graph.add_edge(session_node_id, term_norm, relation="giảng_dạy")
            
        # Concept-to-Concept Co-occurrence Mesh (Nối các khái niệm xuất hiện cùng nhau)
        if len(concepts) > 1:
            for i in range(len(concepts)):
                for j in range(i + 1, min(i + 4, len(concepts))):
                    src_c = concepts[i]
                    tgt_c = concepts[j]
                    course_graph.add_edge(src_c, tgt_c, relation="đồng_xuất_hiện_cùng")
                    global_graph.add_edge(src_c, tgt_c, relation="đồng_xuất_hiện_cùng")

    # Save course partition
    with open(os.path.join(out_dir, "graph.json"), "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(course_graph), f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "knowledge_store.json"), "w", encoding="utf-8") as f:
        json.dump(c_chunks, f, ensure_ascii=False, indent=2)
        
    global_store.extend(c_chunks)
    print(f"Course {cid} ({c_title[:35]}...): {course_graph.number_of_nodes()} nodes, {course_graph.number_of_edges()} edges, {len(c_chunks)} chunks.")

# 3. Add Cross-Course Bridges to global graph
for src_cid, tgt_cid, src_concept, rel, tgt_concept in CROSS_BRIDGES:
    src_title = COURSE_META[src_cid]["title"]
    tgt_title = COURSE_META[tgt_cid]["title"]
    global_graph.add_node(src_concept, type="BridgeConcept", course=src_title)
    global_graph.add_node(tgt_concept, type="BridgeConcept", course=tgt_title)
    global_graph.add_edge(src_concept, tgt_concept, relation=rel)

# Save Master LMS partition
lms_out_dir = os.path.join(GRAPHRAG_DIR, "lms")
os.makedirs(lms_out_dir, exist_ok=True)
with open(os.path.join(lms_out_dir, "graph.json"), "w", encoding="utf-8") as f:
    json.dump(nx.node_link_data(global_graph), f, ensure_ascii=False, indent=2)
with open(os.path.join(lms_out_dir, "knowledge_store.json"), "w", encoding="utf-8") as f:
    json.dump(global_store, f, ensure_ascii=False, indent=2)

print("="*60)
print(f"Master LMS Knowledge Base Rebuilt:")
print(f"- Total Nodes: {global_graph.number_of_nodes()}")
print(f"- Total Edges (Mesh Interconnected): {global_graph.number_of_edges()}")
print(f"- Total Ingested Chunks (Train + Val): {len(global_store)}")
print(f"- Test Chunks (Holdout): 123 chunks STRICTLY EXCLUDED.")
print("="*60)
