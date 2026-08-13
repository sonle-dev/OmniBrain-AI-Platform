import os
import json
import logging
from typing import Optional, List, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)

class NetworkXGraphRAG:
    """
    A pure Python local-search GraphRAG implementation using networkx and LiteLLM.
    Used as a fallback when nano-graphrag (or its dependency graspologic) cannot be installed.
    """
    def __init__(self, working_dir: str):
        self.working_dir = working_dir
        self.graph_path = os.path.join(working_dir, "fallback_graph.json")
        os.makedirs(working_dir, exist_ok=True)
        
        import networkx as nx
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.graph = nx.node_link_graph(data)
                logger.info(f"Loaded existing fallback GraphRAG with {self.graph.number_of_nodes()} nodes.")
            except Exception as e:
                logger.error(f"Failed to load fallback graph: {e}. Creating new graph.")
                self.graph = nx.MultiDiGraph()
        else:
            self.graph = nx.MultiDiGraph()

    def _save_graph(self):
        import networkx as nx
        try:
            data = nx.node_link_data(self.graph)
            with open(self.graph_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save fallback graph: {e}")

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
        Extracts key entities from query, performs local-search in the NetworkX graph,
        and returns matching relationships as text context.
        """
        if self.graph.number_of_nodes() == 0:
            return ""

        from app.services.llm_factory import LLMFactory

        # Step 1: Extract keywords / entities from the user query
        extract_prompt = (
            "Trích xuất các thực thể hoặc từ khóa chính xuất hiện trong câu hỏi sau dưới dạng JSON list của các chuỗi.\n"
            f"CÂU HỎI: \"{query}\"\n"
            "Ví dụ: [\"FastAPI\", \"OmniBrain\"]\n"
            "Chỉ trả về JSON list hợp lệ. Nếu không trích xuất được gì, trả về []."
        )

        def parser_fn(x: str):
            cleaned = x.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)

        try:
            keywords, _ = await LLMFactory.chat_completion_with_self_correction(
                messages=[{"role": "user", "content": extract_prompt}],
                parser_fn=parser_fn,
                temperature=0.1
            )
        except Exception:
            keywords = []

        # If LLM extraction failed, use words in query as simple match keys
        if not keywords:
            keywords = [word.strip(",.!?\"'") for word in query.split() if len(word) > 3]

        # Step 2: Search graph for matching nodes
        matched_nodes = []
        all_nodes = list(self.graph.nodes)
        
        for kw in keywords:
            kw_lower = kw.lower()
            for node in all_nodes:
                if kw_lower in node.lower() or node.lower() in kw_lower:
                    if node not in matched_nodes:
                        matched_nodes.append(node)

        if not matched_nodes:
            return ""

        # Step 3: Retrieve 1-hop subgraph relationships
        subgraph_triples = []
        for node in matched_nodes:
            # Outgoing edges
            if node in self.graph:
                for neighbor in self.graph.successors(node):
                    edge_data = self.graph.get_edge_data(node, neighbor)
                    for key in edge_data:
                        rel = edge_data[key].get("relation", "liên kết")
                        subgraph_triples.append(f"- \"{node}\" --({rel})-> \"{neighbor}\"")
                # Incoming edges
                for predecessor in self.graph.predecessors(node):
                    edge_data = self.graph.get_edge_data(predecessor, node)
                    for key in edge_data:
                        rel = edge_data[key].get("relation", "liên kết")
                        subgraph_triples.append(f"- \"{predecessor}\" --({rel})-> \"{node}\"")

        # Deduplicate
        subgraph_triples = list(set(subgraph_triples))
        
        # Format context
        if subgraph_triples:
            context_text = "Các thông tin và mối quan hệ tìm thấy trong tài liệu:\n" + "\n".join(subgraph_triples[:40])
            return context_text
        return ""


class GraphRAGService:
    def __init__(self):
        self._rag_instance = None
        self._initialized = False
        self._is_fallback = False

    def _lazy_init(self):
        if self._initialized:
            return

        # Attempt to initialize nano-graphrag
        try:
            from nano_graphrag import GraphRAG
            from nano_graphrag._utils import wrap_embedding_func_with_attrs
            import numpy as np
            import litellm
            
            os.makedirs(settings.GRAPHRAG_WORKING_DIR, exist_ok=True)

            @wrap_embedding_func_with_attrs(
                embedding_dim=1536 if "openai" in settings.DEFAULT_CHAT_MODEL else 768,
                max_token_size=8192,
            )
            async def custom_embedding(texts: list[str]) -> np.ndarray:
                try:
                    embed_model = "text-embedding-3-small" if "openai" in settings.DEFAULT_CHAT_MODEL else "gemini/text-embedding-004"
                    response = await litellm.aembedding(
                        model=embed_model,
                        input=texts
                    )
                    embeddings = [item["embedding"] for item in response.data]
                    return np.array(embeddings, dtype=np.float32)
                except Exception as e:
                    logger.error(f"Error in GraphRAG custom embedding: {e}")
                    response = await litellm.aembedding(
                        model="openai/text-embedding-ada-002",
                        input=texts
                    )
                    embeddings = [item["embedding"] for item in response.data]
                    return np.array(embeddings, dtype=np.float32)

            async def custom_llm(
                prompt: str,
                system_prompt: Optional[str] = None,
                history_messages: list = [],
                **kwargs
            ) -> str:
                from app.services.llm_factory import LLMFactory
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                for msg in history_messages:
                    messages.append(msg)
                messages.append({"role": "user", "content": prompt})

                try:
                    result = await LLMFactory.chat_completion(
                        messages=messages,
                        temperature=kwargs.get("temperature", 0.1),
                        max_tokens=kwargs.get("max_tokens", 2048)
                    )
                    return result["text"]
                except Exception as e:
                    logger.error(f"Error in GraphRAG custom LLM call: {e}")
                    raise e

            self._rag_instance = GraphRAG(
                working_dir=settings.GRAPHRAG_WORKING_DIR,
                embedding_func=custom_embedding,
                llm_func=custom_llm
            )
            self._is_fallback = False
            self._initialized = True
            logger.info("GraphRAG Engine (nano-graphrag) lazy initialized successfully.")
            
        except Exception as e:
            logger.warning(
                f"Failed to load nano-graphrag: {e}. "
                "Activating self-healing fallback: Pure-Python NetworkX GraphRAG."
            )
            # Use local fallback GraphRAG
            self._rag_instance = NetworkXGraphRAG(working_dir=settings.GRAPHRAG_WORKING_DIR)
            self._is_fallback = True
            self._initialized = True
            logger.info("Fallback NetworkX GraphRAG lazy initialized successfully.")

    def extract_text(self, file_path: str) -> str:
        """
        Extracts plain text from standard files, PDFs, or PPTX slides.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            try:
                import pdfplumber
                text_list = []
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text_list.append(extracted)
                return "\n".join(text_list)
            except ImportError:
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

    async def insert_document(self, file_path: str):
        """
        Ingests a document (PDF, Slide, TXT) into the Knowledge Graph database.
        """
        self._lazy_init()
        text = self.extract_text(file_path)
        if not text.strip():
            raise ValueError("The document contains no readable text.")
        
        logger.info(f"Ingesting text ({len(text)} chars) from {file_path} into GraphRAG...")
        await self._rag_instance.ainsert(text)
        logger.info("Document ingested successfully.")

    async def query_graph(self, query: str, mode: str = "local") -> str:
        """
        Queries the Knowledge Graph. Adheres to Strict Grounding rule.
        """
        self._lazy_init()
        from app.services.llm_factory import LLMFactory

        logger.info(f"Querying GraphRAG (Fallback: {self._is_fallback}) in {mode} mode: '{query}'")

        try:
            if self._is_fallback:
                context = await self._rag_instance.aquery(query, param=None)
            else:
                from nano_graphrag import QueryParam
                context = await self._rag_instance.aquery(
                    query,
                    param=QueryParam(mode=mode, only_need_context=True)
                )
        except Exception as e:
            logger.error(f"Failed to query GraphRAG context: {e}")
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
