import os
import json
import logging
from typing import Optional, List, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)

def remove_vietnamese_accents(s: str) -> str:
    """Converts accented Vietnamese text to base ASCII ('quản lý' -> 'quan ly')."""
    if not s:
        return ""
    s = str(s).replace('đ', 'd').replace('Đ', 'D')
    import unicodedata
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower()

def decode_telex_typos(s: str) -> str:
    """Decodes common typing errors when user types Telex on English IME."""
    import re
    t = str(s).lower()
    t = re.sub(r'aa|aw', 'a', t)
    t = re.sub(r'ee', 'e', t)
    t = re.sub(r'oo|ow', 'o', t)
    t = re.sub(r'uw', 'u', t)
    t = re.sub(r'dd', 'd', t)
    t = re.sub(r'[sfrxj](?=\b|\s|[0-9])', '', t)
    return t

class NetworkXGraphRAG:
    """
    A pure Python local-search GraphRAG implementation using networkx and LiteLLM.
    Used as a fallback when nano-graphrag (or its dependency graspologic) cannot be installed.
    """
    def __init__(self, working_dir: str):
        self.working_dir = working_dir
        self.graph_path = os.path.join(working_dir, "graph.json")
        self.store_path = os.path.join(working_dir, "knowledge_store.json")
        legacy_path = os.path.join(working_dir, "fallback_graph.json")
        os.makedirs(working_dir, exist_ok=True)
        
        import networkx as nx
        load_path = self.graph_path if os.path.exists(self.graph_path) else legacy_path
        if os.path.exists(load_path):
            try:
                with open(load_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.graph = nx.node_link_graph(data)
                logger.info(f"Loaded existing GraphRAG at {load_path} with {self.graph.number_of_nodes()} nodes.")
            except Exception as e:
                logger.error(f"Failed to load graph at {load_path}: {e}. Creating new graph.")
                self.graph = nx.MultiDiGraph()
        else:
            self.graph = nx.MultiDiGraph()

        self.knowledge_store = []
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    self.knowledge_store = json.load(f)
                logger.info(f"Loaded knowledge store at {self.store_path} with {len(self.knowledge_store)} chunks.")
            except Exception as e:
                logger.warning(f"Could not load knowledge store: {e}")

    def _save_graph(self):
        import networkx as nx
        try:
            data = nx.node_link_data(self.graph)
            with open(self.graph_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save fallback graph: {e}")

    def reset(self):
        """Clears in-memory graph and removes on-disk graph files for this partition."""
        import networkx as nx
        self.graph = nx.MultiDiGraph()
        for p in [self.graph_path, os.path.join(self.working_dir, "fallback_graph.json")]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    def get_stats(self) -> Dict[str, Any]:
        """Returns node and edge metrics for this graph partition."""
        return {
            "nodes_count": self.graph.number_of_nodes(),
            "edges_count": self.graph.number_of_edges()
        }

    async def ainsert(self, text: str):
        """
        Chunks the text, uses LLM to extract entities and relations,
        and saves them into the NetworkX graph.
        """
        # Chunk text
        chunk_size = 800
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        from app.services.llm_factory import LLMFactory
        import networkx as nx

        logger.info(f"Extracting KG triples from {len(chunks)} chunks using LLM...")

        for idx, chunk in enumerate(chunks):
            prompt = (
                "Nhiệm vụ của bạn là trích xuất các Thực thể (Entities) và Mối quan hệ (Relationships) "
                "giữa chúng từ đoạn văn bản dưới đây.\n\n"
                f"ĐOẠN VĂN BẢN:\n---\n{chunk}\n---\n\n"
                "Hãy trả về kết quả dưới dạng một JSON list của các đối tượng chứa: "
                "\"source\" (Tên thực thể 1), \"relation\" (Mối quan hệ giữa chúng), \"target\" (Tên thực thể 2).\n"
                "Ví dụ:\n"
                "[\n"
                "  {\"source\": \"FastAPI\", \"relation\": \"là\", \"target\": \"Web Framework\"},\n"
                "  {\"source\": \"OmniBrain\", \"relation\": \"sử dụng\", \"target\": \"FastAPI\"}\n"
                "]\n"
                "Chỉ trả về JSON hợp lệ. Nếu không có mối quan hệ nào, trả về []."
            )
            
            # Use self-correction to guarantee valid JSON
            def parser_fn(x: str):
                # Clean up any trailing/leading markdowns
                cleaned = x.strip().replace("```json", "").replace("```", "").strip()
                return json.loads(cleaned)

            try:
                triples, _ = await LLMFactory.chat_completion_with_self_correction(
                    messages=[{"role": "user", "content": prompt}],
                    parser_fn=parser_fn,
                    temperature=0.1
                )
                
                # Add to graph
                for item in triples:
                    src = item.get("source", "").strip()
                    rel = item.get("relation", "").strip()
                    tgt = item.get("target", "").strip()
                    
                    if src and tgt and rel:
                        self.graph.add_node(src)
                        self.graph.add_node(tgt)
                        self.graph.add_edge(src, tgt, relation=rel)
                
            except Exception as e:
                logger.error(f"Failed to extract triples for chunk {idx}: {e}")
                continue

        self._save_graph()
        logger.info(f"Fallback graph insertion complete. Graph now has {self.graph.number_of_nodes()} nodes.")

    async def aquery(self, query: str, param=None) -> str:
        """
        Performs accent-resilient, telex-aware, n-gram hybrid search across the
        Knowledge Graph and knowledge store chunks, returning grounded lecture context.
        """
        if self.graph.number_of_nodes() == 0 and not self.knowledge_store:
            return ""

        import re

        clean_q = query.strip()
        lower_q = clean_q.lower()
        norm_q = remove_vietnamese_accents(clean_q)
        telex_q = decode_telex_typos(norm_q)

        # Step 1: Extract multi-variant search tokens (words >= 2 chars)
        candidate_tokens = set()
        for text in [lower_q, norm_q, telex_q]:
            for w in re.findall(r'[\w\-]+', text):
                if len(w) >= 2:
                    candidate_tokens.add(w)
                    candidate_tokens.add(remove_vietnamese_accents(w))

        # Bigram / phrase tokens (e.g. "quan ly", "mo hinh", "phan tich", "chuyen sau", "b tree", "spring boot")
        norm_words = [w for w in re.findall(r'[\w\-]+', norm_q) if len(w) >= 2]
        candidate_bigrams = []
        for i in range(len(norm_words) - 1):
            candidate_bigrams.append(f"{norm_words[i]} {norm_words[i+1]}")
            candidate_bigrams.append(f"{norm_words[i]}{norm_words[i+1]}")

        # Step 2: Search graph nodes using accent-normalized matching
        matched_nodes = []
        all_nodes = list(self.graph.nodes)
        
        for node in all_nodes:
            node_norm = remove_vietnamese_accents(node)
            for bg in candidate_bigrams:
                if bg in node_norm or node_norm in bg:
                    if node not in matched_nodes:
                        matched_nodes.append(node)
            for tok in candidate_tokens:
                if len(tok) >= 3 and (tok == node_norm or tok in node_norm):
                    if node not in matched_nodes:
                        matched_nodes.append(node)

        # Step 3: Retrieve 1-hop subgraph relationships
        subgraph_triples = []
        for node in matched_nodes[:20]:
            if node in self.graph:
                for neighbor in self.graph.successors(node):
                    edge_data = self.graph.get_edge_data(node, neighbor)
                    for key in edge_data:
                        rel = edge_data[key].get("relation", "liên kết")
                        subgraph_triples.append(f"- \"{node}\" --({rel})-> \"{neighbor}\"")
                for predecessor in self.graph.predecessors(node):
                    edge_data = self.graph.get_edge_data(predecessor, node)
                    for key in edge_data:
                        rel = edge_data[key].get("relation", "liên kết")
                        subgraph_triples.append(f"- \"{predecessor}\" --({rel})-> \"{node}\"")

        subgraph_triples = list(set(subgraph_triples))[:25]

        # Step 4: Search knowledge chunks with dual-layer (word + phrase) scoring
        scored_chunks = []
        if self.knowledge_store:
            for chunk in self.knowledge_store:
                c_text = chunk.get("text", "")
                c_text_lower = c_text.lower()
                c_text_norm = remove_vietnamese_accents(c_text)
                score = 0
                
                # Check bigrams / phrases (very high relevance!)
                for bg in candidate_bigrams:
                    if bg in c_text_norm:
                        score += 15
                
                # Check individual tokens
                for tok in candidate_tokens:
                    if tok in c_text_norm:
                        score += 2
                    if tok in c_text_lower:
                        score += 3
                    if any(tok in remove_vietnamese_accents(str(k)) for k in chunk.get("keywords", [])):
                        score += 6
                
                # Boost if session matches (e.g. Buổi 1, Buổi 2)
                for s_num in range(1, 10):
                    if f"buoi {s_num}" in norm_q or f"bai {s_num}" in norm_q:
                        if chunk.get("session") == s_num:
                            score += 20

                if score > 0:
                    scored_chunks.append((score, chunk))

            scored_chunks.sort(key=lambda x: x[0], reverse=True)

        top_chunks = [item[1] for item in scored_chunks[:3]]

        # Step 5: Format grounded context
        sections = []
        if subgraph_triples:
            sections.append("CẤU TRÚC TRI THỨC VÀ MỐI QUAN HỆ (KNOWLEDGE GRAPH):\n" + "\n".join(subgraph_triples))

        if top_chunks:
            chunk_texts = []
            for c in top_chunks:
                header = f"[{c.get('courseTitle', '')} - Buổi {c.get('session', '')} (Trang {c.get('page', '')})]"
                chunk_texts.append(f"{header}\n{c.get('text', '')[:1400]}")
            sections.append("NỘI DUNG CHI TIẾT TỪ GIÁO TRÌNH BÀI GIẢNG CHÍNH THỨC:\n\n" + "\n\n---\n\n".join(chunk_texts))

        return "\n\n".join(sections) if sections else ""


class GraphRAGService:
    def __init__(self):
        self._engines: Dict[str, NetworkXGraphRAG] = {}

    def _clean_namespace(self, namespace: Optional[str]) -> str:
        if not namespace or not str(namespace).strip():
            return "lms"
        cleaned = "".join(c for c in str(namespace).strip().lower() if c.isalnum() or c in ("_", "-"))
        return cleaned or "lms"

    def _get_engine(self, namespace: Optional[str] = "lms") -> NetworkXGraphRAG:
        ns = self._clean_namespace(namespace)
        if ns not in self._engines:
            ns_dir = os.path.join(settings.GRAPHRAG_WORKING_DIR, ns)
            legacy_root_graph = os.path.join(settings.GRAPHRAG_WORKING_DIR, "fallback_graph.json")
            target_graph = os.path.join(ns_dir, "graph.json")
            if not os.path.exists(target_graph) and os.path.exists(legacy_root_graph) and ns in ("lms", "default"):
                os.makedirs(ns_dir, exist_ok=True)
                try:
                    import shutil
                    shutil.copy2(legacy_root_graph, target_graph)
                    logger.info("Migrated legacy root graph to namespace '%s'", ns)
                except Exception as e:
                    logger.warning("Could not copy legacy graph: %s", e)

            self._engines[ns] = NetworkXGraphRAG(working_dir=ns_dir)
            logger.info("Initialized GraphRAG engine for namespace '%s' in %s", ns, ns_dir)
        return self._engines[ns]

    def list_namespaces(self) -> List[Dict[str, Any]]:
        """Lists all existing knowledge base partitions with node and edge metrics."""
        results = []
        os.makedirs(settings.GRAPHRAG_WORKING_DIR, exist_ok=True)
        detected_ns = set(self._engines.keys())
        detected_ns.add("lms")
        for entry in os.listdir(settings.GRAPHRAG_WORKING_DIR):
            full_path = os.path.join(settings.GRAPHRAG_WORKING_DIR, entry)
            if os.path.isdir(full_path) and entry not in ("temp_uploads", "temp_audio"):
                detected_ns.add(entry)
        
        for ns in sorted(detected_ns):
            engine = self._get_engine(ns)
            stats = engine.get_stats()
            results.append({
                "namespace": ns,
                "nodes_count": stats["nodes_count"],
                "edges_count": stats["edges_count"]
            })
        return results

    def reset_namespace(self, namespace: str = "lms") -> Dict[str, Any]:
        """Resets a single knowledge base partition without impacting other partitions."""
        ns = self._clean_namespace(namespace)
        engine = self._get_engine(ns)
        engine.reset()
        logger.info("Reset GraphRAG namespace '%s'", ns)
        return {
            "status": "success",
            "namespace": ns,
            "message": f"Knowledge base partition '{ns}' has been reset successfully."
        }

    def extract_text(self, file_path: str) -> str:
        """
        Extracts plain text from standard files, PDFs, or PPTX slides.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            try:
                import pypdfium2 as pdfium
                doc = pdfium.PdfDocument(file_path)
                text_list = []
                for page in doc:
                    textpage = page.get_textpage()
                    t = textpage.get_text_range()
                    if t and t.strip():
                        text_list.append(t.strip())
                if text_list:
                    return "\n\n".join(text_list)
            except Exception as e:
                logger.warning(f"pypdfium2 extraction failed, trying fallback: {e}")

            try:
                import pdfplumber
                text_list = []
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text_list.append(extracted)
                return "\n".join(text_list)
            except Exception:
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(file_path)
                    return "\n".join([page.extract_text() or "" for page in reader.pages])
                except Exception as e:
                    logger.error(f"PDF extraction failure: {e}")
                    raise RuntimeError(f"Could not extract text from PDF: {e}")
        elif ext in [".pptx", ".ppt"]:
            try:
                from pptx import Presentation
                prs = Presentation(file_path)
                text_runs = []
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text") and shape.text:
                            text_runs.append(shape.text)
                return "\n".join(text_runs)
            except ImportError:
                raise RuntimeError("python-pptx package required to parse Slides")
            except Exception as e:
                logger.error(f"Slides extraction failure: {e}")
                raise RuntimeError(f"Could not extract text from Slides: {e}")
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    async def insert_document(self, file_path: str, namespace: str = "lms"):
        """
        Ingests a document (PDF, Slide, TXT) into the specified Knowledge Graph namespace partition.
        """
        ns = self._clean_namespace(namespace)
        engine = self._get_engine(ns)
        text = self.extract_text(file_path)
        if not text.strip():
            raise ValueError("The document contains no readable text.")
        
        logger.info(f"Ingesting text ({len(text)} chars) from {file_path} into GraphRAG namespace '{ns}'...")
        await engine.ainsert(text)
        logger.info(f"Document ingested successfully into namespace '{ns}'.")

    async def query_graph(self, query: str, mode: str = "local", namespace: str = "lms") -> str:
        """
        Queries the Knowledge Graph partition for the given namespace. Strictly respects local context.
        """
        ns = self._clean_namespace(namespace)
        engine = self._get_engine(ns)
        from app.services.llm_factory import LLMFactory

        logger.info(f"Querying GraphRAG namespace '{ns}' in {mode} mode: '{query}'")

        try:
            context = await engine.aquery(query, param=None)
        except Exception as e:
            logger.error(f"Failed to query GraphRAG context in namespace '{ns}': {e}")
            context = ""

        # Programmatic grounding check for empty or very short context
        if not context or len(context.strip()) < 20:
            return "Dựa trên tài liệu được cung cấp, tôi không tìm thấy thông tin này."

        # Strict Grounding System Prompt
        system_prompt = (
            "Bạn là một trợ lý AI chuyên nghiệp. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng "
            "dựa trên thông tin ngữ cảnh (Knowledge Graph) được cung cấp dưới đây.\n\n"
            f"DƯỚI ĐÂY LÀ NGỮ CẢNH TỪ KNOWLEDGE GRAPH:\n---\n{context}\n---\n\n"
            "QUY TẮC RÀNG BUỘC CỰC KỲ NGHIÊM NGẶT (STRICT GROUNDING RULE):\n"
            "1. CHỈ sử dụng thông tin có mặt TRONG NGỮ CẢNH trên để trả lời.\n"
            "2. Không được sử dụng bất kỳ kiến thức bên ngoài nào khác.\n"
            "3. Tuyệt đối không được suy diễn, phỏng đoán, hoặc thêm bớt thông tin không có trong ngữ cảnh.\n"
            "4. Nếu câu trả lời không thể được tìm thấy trực tiếp trong ngữ cảnh, hoặc thông tin bị thiếu, "
            "hoặc câu hỏi nằm ngoài phạm vi ngữ cảnh, bạn BẮT BUỘC phải trả lời chính xác và duy nhất cụm từ sau: "
            "\"Dựa trên tài liệu được cung cấp, tôi không tìm thấy thông tin này.\" "
            "(không thêm bớt bất kỳ từ ngữ hay dấu câu nào khác, không kèm theo lý do)."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        try:
            result = await LLMFactory.chat_completion(
                messages=messages,
                temperature=0.1
            )
            ans = result["text"].strip()
            return ans
        except Exception as e:
            logger.error(f"LLM call during GraphRAG query failed: {e}")
            return "Dựa trên tài liệu được cung cấp, tôi không tìm thấy thông tin này."

# Singleton Instance
graph_rag_service = GraphRAGService()
