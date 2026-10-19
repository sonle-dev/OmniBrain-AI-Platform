import os
import re
import json
import logging
import networkx as nx
import pypdfium2 as pdfium

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IngestionEngine")

BASE_DIR = r"D:\Data Train"
GRAPHRAG_BASE = r"D:\OmniBrain AI Platform\graphrag_data"

COURSE_MAP = {
    "Kiến trúc & Tối ưu Cơ sở Dữ liệu PostgreSQL Enterprise": {
        "id": "c-102",
        "code": "CNTT.K22B.D1.K2.N02",
        "title": "Kiến trúc & Tối ưu Cơ sở Dữ liệu PostgreSQL Enterprise",
        "instructor": "ThS. Nguyễn Hoàng Nam"
    },
    "Lập trình Enterprise với Java 1721 & Spring Boot 3": {
        "id": "c-101",
        "code": "CNTT.K22B.D1.K2.N01",
        "title": "Lập trình Enterprise với Java 17/21 & Spring Boot 3",
        "instructor": "PGS. TS. Trần Đức Minh"
    },
    "Phương pháp phát triển phần mềm hướng đối tượng-2-25": {
        "id": "c-103",
        "code": "CNTT.K22B.D1.K2.N03",
        "title": "Phương pháp phát triển phần mềm hướng đối tượng-2-25",
        "instructor": "TS. Trần Thị Mai"
    },
    "Quản lý dự án CNTT-2-25": {
        "id": "c-104",
        "code": "CNTT.K22B.D1.K2.N04",
        "title": "Quản lý dự án CNTT-2-25",
        "instructor": "TS. Phạm Đình Lâm"
    },
    "Quản trị hệ thống-2-25": {
        "id": "c-105",
        "code": "CNTT.K22B.D1.K2.N05",
        "title": "Quản trị hệ thống-2-25",
        "instructor": "PGS. TS. Nguyễn Văn Học"
    }
}

KEY_TERM_PATTERNS = [
    r'\b(?:MVCC|Postmaster|WAL|Checkpointer|Shared Buffers|Work Mem|Maintenance Work Mem|HikariCP|B-Tree|GiST|GIN|BRIN|EXPLAIN ANALYZE|Vacuum|Autovacuum|Partitioning|Streaming Replication|PITR|pg_dump|pg_basebackup|Transaction Isolation|Locking|Serializable|Repeatable Read|Read Committed)\b',
    r'\b(?:Java 17|Java 21|Spring Boot 3|Spring Security 6|Stateless JWT|JWT Filter|Clean Architecture|Microservices|Spring Cloud|Hibernate 6|Spring Data JPA|MapStruct|Records|Virtual Threads|Pattern Matching|Redis|RabbitMQ|Kafka|Docker Compose|CI/CD|IoC|Dependency Injection|Bean Lifecycle)\b',
    r'\b(?:OOP|Tính đóng gói|Tính kế thừa|Tính đa hình|Tính trừu tượng|Encapsulation|Inheritance|Polymorphism|Abstraction|SOLID|Single Responsibility|Open Closed|Liskov Substitution|Interface Segregation|Dependency Inversion|UML|Use Case|Class Diagram|Sequence Diagram|Activity Diagram|State Machine|Design Patterns|GoF|Singleton|Factory|Observer|Strategy|MVC|JUnit 5|Mockito)\b',
    r'\b(?:Project Charter|Stakeholder|WBS|Work Breakdown Structure|Critical Path Method|CPM|Agile|Scrum|Sprint|Backlog|Burndown Chart|Risk Management|Risk Register|Earned Value Management|EVM|PV|EV|AC|CPI|SPI|QA/QC|SLA|KPI|Procurement|Nghiệm thu)\b',
    r'\b(?:Linux|Ubuntu|RHEL|Kernel|Systemd|Systemctl|Journalctl|Network Configuration|IP Routing|Firewall|UFW|Iptables|LVM|Logical Volume|Ext4|XFS|Process Management|Cron|Crontab|Logrotate|Prometheus|Grafana|Backup Strategy|Disaster Recovery|DNS|BIND9|Nginx|Reverse Proxy|Load Balancing|SSL|TLS|Let\'s Encrypt|Sudoers|PAM|SSH Keypair)\b'
]
COMBINED_TERM_REGEX = re.compile('|'.join(KEY_TERM_PATTERNS), re.IGNORECASE)

def extract_session_number(filename: str) -> int:
    m = re.search(r'buoi[_\s\-]*(\d+)', filename, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 1

def clean_text(text: str) -> str:
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def process_course(folder_name: str, meta: dict, global_graph: nx.MultiDiGraph, global_store: list):
    course_id = meta["id"]
    course_title = meta["title"]
    instructor = meta["instructor"]
    code = meta["code"]
    
    course_dir = os.path.join(BASE_DIR, folder_name)
    out_dir = os.path.join(GRAPHRAG_BASE, course_id)
    os.makedirs(out_dir, exist_ok=True)
    
    course_graph = nx.MultiDiGraph()
    course_store = []
    
    # Root course node
    course_graph.add_node(course_title, type="Course", id=course_id, code=code, instructor=instructor)
    global_graph.add_node(course_title, type="Course", id=course_id, code=code, instructor=instructor)
    
    pdf_files = sorted([f for f in os.listdir(course_dir) if f.lower().endswith(".pdf")])
    logger.info(f"Processing course: {course_title} ({len(pdf_files)} PDF files)")
    
    for fn in pdf_files:
        session_num = extract_session_number(fn)
        pdf_path = os.path.join(course_dir, fn)
        doc = pdfium.PdfDocument(pdf_path)
        page_count = len(doc)
        
        # Read all pages
        pages_content = []
        for p_idx in range(page_count):
            p_text = clean_text(doc[p_idx].get_textpage().get_text_range())
            if p_text:
                pages_content.append((p_idx + 1, p_text))
                
        if not pages_content:
            continue
            
        first_page_text = pages_content[0][1]
        session_title = f"Buổi {session_num}"
        for line in first_page_text.splitlines():
            clean_l = line.strip()
            if re.search(r'BUOI\s*\d+:', clean_l, re.IGNORECASE):
                session_title = clean_l
                break
            elif re.search(r'Tai lieu Buổi\s*\d+:', clean_l, re.IGNORECASE):
                session_title = clean_l
                break
                
        session_node_id = f"{course_title} - Buổi {session_num}"
        session_desc = f"{session_title} ({page_count} trang). Giảng viên: {instructor}."
        
        # Graph nodes
        course_graph.add_node(session_node_id, type="Session", number=session_num, title=session_title, course=course_title)
        course_graph.add_edge(course_title, session_node_id, relation="chương_trình_học")
        
        global_graph.add_node(session_node_id, type="Session", number=session_num, title=session_title, course=course_title)
        global_graph.add_edge(course_title, session_node_id, relation="chương_trình_học")
        
        # Process pages into structured chunks and extract entities
        chunk_size_chars = 1200
        current_chunk = []
        current_len = 0
        chunk_page_start = 1
        
        for p_num, p_text in pages_content:
            paragraphs = [p.strip() for p in p_text.split('\n\n') if p.strip()]
            for para in paragraphs:
                current_chunk.append(para)
                current_len += len(para)
                
                if current_len >= chunk_size_chars:
                    full_chunk_text = "\n\n".join(current_chunk)
                    chunk_id = f"{course_id}-b{session_num}-p{chunk_page_start}-{len(course_store)+1}"
                    
                    # Extract key terms found in this chunk
                    matched_terms = list(set(m.group(0).strip() for m in COMBINED_TERM_REGEX.finditer(full_chunk_text)))
                    
                    item = {
                        "id": chunk_id,
                        "courseId": course_id,
                        "courseCode": code,
                        "courseTitle": course_title,
                        "session": session_num,
                        "sessionTitle": session_title,
                        "fileName": fn,
                        "page": chunk_page_start,
                        "text": full_chunk_text,
                        "keywords": matched_terms
                    }
                    course_store.append(item)
                    global_store.append(item)
                    
                    # Add matched terms into graph
                    for term in matched_terms:
                        term_norm = term.strip().title() if len(term) > 4 else term.strip().upper()
                        course_graph.add_node(term_norm, type="Concept", course=course_title)
                        course_graph.add_edge(session_node_id, term_norm, relation="giảng_dạy", page=chunk_page_start)
                        
                        global_graph.add_node(term_norm, type="Concept", course=course_title)
                        global_graph.add_edge(session_node_id, term_norm, relation="giảng_dạy", page=chunk_page_start)
                    
                    current_chunk = []
                    current_len = 0
                    chunk_page_start = p_num
                    
        # Flush remaining chunk if any
        if current_chunk:
            full_chunk_text = "\n\n".join(current_chunk)
            chunk_id = f"{course_id}-b{session_num}-p{chunk_page_start}-{len(course_store)+1}"
            matched_terms = list(set(m.group(0).strip() for m in COMBINED_TERM_REGEX.finditer(full_chunk_text)))
            item = {
                "id": chunk_id,
                "courseId": course_id,
                "courseCode": code,
                "courseTitle": course_title,
                "session": session_num,
                "sessionTitle": session_title,
                "fileName": fn,
                "page": chunk_page_start,
                "text": full_chunk_text,
                "keywords": matched_terms
            }
            course_store.append(item)
            global_store.append(item)
            
            for term in matched_terms:
                term_norm = term.strip().title() if len(term) > 4 else term.strip().upper()
                course_graph.add_node(term_norm, type="Concept", course=course_title)
                course_graph.add_edge(session_node_id, term_norm, relation="giảng_dạy", page=chunk_page_start)
                
                global_graph.add_node(term_norm, type="Concept", course=course_title)
                global_graph.add_edge(session_node_id, term_norm, relation="giảng_dạy", page=chunk_page_start)

    # Save partition graph and store
    graph_data = nx.node_link_data(course_graph)
    with open(os.path.join(out_dir, "graph.json"), "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
    with open(os.path.join(out_dir, "knowledge_store.json"), "w", encoding="utf-8") as f:
        json.dump(course_store, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Course '{course_id}' completed: {course_graph.number_of_nodes()} nodes, {course_graph.number_of_edges()} edges, {len(course_store)} text chunks.")

def main():
    logger.info("STARTING BULK INGESTION OF ALL 45 PDF FILES INTO OMNIBRAIN KNOWLEDGE BASE...")
    
    global_graph = nx.MultiDiGraph()
    global_store = []
    
    for folder, meta in COURSE_MAP.items():
        process_course(folder, meta, global_graph, global_store)
        
    # Save unified LMS partition
    lms_dir = os.path.join(GRAPHRAG_BASE, "lms")
    os.makedirs(lms_dir, exist_ok=True)
    
    global_graph_data = nx.node_link_data(global_graph)
    with open(os.path.join(lms_dir, "graph.json"), "w", encoding="utf-8") as f:
        json.dump(global_graph_data, f, ensure_ascii=False, indent=2)
        
    with open(os.path.join(lms_dir, "knowledge_store.json"), "w", encoding="utf-8") as f:
        json.dump(global_store, f, ensure_ascii=False, indent=2)
        
    logger.info(f"=== GLOBAL LMS KNOWLEDGE BASE SAVED ===")
    logger.info(f"Total Graph Nodes: {global_graph.number_of_nodes()}")
    logger.info(f"Total Graph Edges: {global_graph.number_of_edges()}")
    logger.info(f"Total Indexed Chunks: {len(global_store)}")
    logger.info("ALL 45 COURSE TEXTBOOKS HAVE BEEN INGESTED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
