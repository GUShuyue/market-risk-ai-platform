"""
Retriever Agent — 混合检索与RAG (Day 3-4 实现)

负责:
  1. 将转录文本分块 (chunking)
  2. 使用 DashScope text-embedding-v4 API 生成 Embedding 向量
  3. 存入 ChromaDB 向量数据库
  4. 混合检索: 语义检索 + BM25关键词检索 → 各取 top_k → RRF 融合去重
  5. 重排序(必须): DashScope gte-rerank 精排 → 本地 BM25 回退 → 取 top_k
  6. 构建增强上下文

技术栈:
  - DashScope API: text-embedding-v4 (1024维) + gte-rerank (重排序)
  - ChromaDB: 向量数据库 (语义检索)
  - rank-bm25: BM25 关键词检索 + 本地重排序回退
  - 回退: 本地 BGE 模型 或 纯关键词匹配
"""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from ..config import get_config
from ..models.messages import (
    AgentStatus, TranscriberOutput, RetrieverOutput, RetrievedChunk,
)
from .base import BaseAgent

logger = logging.getLogger(__name__)


class RetrieverAgent(BaseAgent[TranscriberOutput, RetrieverOutput]):
    """
    RAG检索Agent

    输入: TranscriberOutput (包含转录文本 + 分段信息)
    输出: RetrieverOutput (包含检索到的相关段落 + 增强上下文)
    """

    agent_name = "retriever"

    def __init__(self):
        super().__init__()
        self._embedding_model = None
        self._chroma_client = None
        self._collection = None

    def execute(self, input_data: TranscriberOutput) -> RetrieverOutput:
        self.log_start(
            text_length=len(input_data.full_text),
            segments=len(input_data.segments),
        )

        output = RetrieverOutput(
            agent_status=AgentStatus.RUNNING,
            current_text=input_data.full_text,
        )

        try:
            # 1. 文本分块
            chunks = self._chunk_text(
                input_data.full_text,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            self.logger.info(f"  文本分块: {len(chunks)} 个块")

            if not chunks:
                output.agent_status = AgentStatus.SUCCESS
                output.augmented_context = input_data.full_text
                return output

            # 2. 生成 Embedding 并存入 ChromaDB
            try:
                self._embed_and_store(chunks, input_data)
            except Exception as e:
                self.logger.warning(f"  Embedding/ChromaDB 不可用: {e}")
                self.logger.info("  回退到关键词匹配模式")

            # 3. 检索相关段落
            query = self._build_query(input_data)
            retrieved = self._retrieve_similar(query, top_k=self.config.rag_top_k)

            # 4. 构建增强上下文
            augmented = self._build_augmented_context(
                input_data.full_text, retrieved
            )

            output.retrieved_chunks = retrieved
            output.augmented_context = augmented
            output.agent_status = AgentStatus.SUCCESS

            self.log_done(
                chunks=len(chunks),
                retrieved=len(retrieved),
                context_length=len(augmented),
            )

        except Exception as e:
            self.log_error(str(e))
            output.agent_status = AgentStatus.FAILED
            output.error_message = str(e)
            # 失败时至少返回原始文本
            output.augmented_context = input_data.full_text

        return output

    # ================================================================
    # 文本分块
    # ================================================================

    def _chunk_text(
        self, text: str, chunk_size: int = 2000, chunk_overlap: int = 200
    ) -> List[str]:
        """
        将长文本切分为重叠的块

        策略:
          - 优先按段落(\n\n)切分
          - 段落太长则按句子切分
          - 保持 chunk_overlap 确保上下文连贯
        """
        if not text.strip():
            return []

        # 按段落切分
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        current_length = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_words = len(para.split())

            if current_length + para_words > chunk_size:
                # 当前块已满，保存并开始新块
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # 如果单个段落超过块大小，按句子切分
                if para_words > chunk_size:
                    sub_chunks = self._split_long_paragraph(para, chunk_size, chunk_overlap)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                    current_length = 0
                else:
                    # 新块从 overlap 开始
                    if chunks:
                        overlap_text = " ".join(chunks[-1].split()[-chunk_overlap:])
                        current_chunk = overlap_text + "\n" + para
                    else:
                        current_chunk = para
                    current_length = len(current_chunk.split())
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_length += para_words

        # 最后一个块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _split_long_paragraph(
        self, text: str, chunk_size: int, overlap: int
    ) -> List[str]:
        """将超长段落按句子切分"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""
        current_len = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_len = len(sent.split())

            if current_len + sent_len > chunk_size and current:
                chunks.append(current.strip())
                # 保留 overlap
                overlap_words = " ".join(current.split()[-overlap:])
                current = overlap_words + " " + sent
                current_len = len(current.split())
            else:
                current = (current + " " + sent) if current else sent
                current_len += sent_len

        if current.strip():
            chunks.append(current.strip())

        return chunks

    # ================================================================
    # Embedding (DashScope API / 本地BGE)
    # ================================================================

    def _get_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        获取文本的 Embedding 向量

        策略:
          1. 优先 OpenAI Compatible Embedding API (默认 DashScope text-embedding-v4)
          2. 回退本地 BGE 模型 (sentence-transformers)
          3. 都不可用返回 None → 触发关键词回退
        """
        # 策略1: OpenAI Compatible API
        if self.config.embedding_provider == "api" and self.config.embedding_available:
            try:
                return self._api_embed(texts)
            except Exception as e:
                self.logger.warning(f"  Embedding API 失败: {e}")

        # 策略2: 本地 BGE 模型
        if self.config.embedding_provider == "local":
            try:
                return self._local_bge_embed(texts)
            except Exception as e:
                self.logger.warning(f"  本地 BGE 失败: {e}")

        # 策略3: 不可用
        self.logger.warning("  Embedding 不可用，将回退到关键词匹配")
        return None

    def _api_embed(self, texts: List[str]) -> List[List[float]]:
        """通过 OpenAI Compatible API 获取 Embedding"""
        from openai import OpenAI

        self.logger.info(f"  调用 Embedding API: {self.config.embedding_model} ({len(texts)} 条)")

        client = OpenAI(
            api_key=self.config.embedding_api_key,
            base_url=self.config.embedding_base_url,
            timeout=self.config.request_timeout,
        )
        resp = client.embeddings.create(
            model=self.config.embedding_model,
            input=texts,
        )

        embeddings = [item.embedding for item in resp.data]
        self.logger.info(f"  Embedding 完成: {len(embeddings)} 条, dim={len(embeddings[0])}")
        return embeddings

    def _local_bge_embed(self, texts: List[str]) -> List[List[float]]:
        """通过本地 BGE 模型获取 Embedding"""
        from sentence_transformers import SentenceTransformer

        if self._embedding_model is None:
            self.logger.info(f"加载本地 Embedding 模型: {self.config.embedding_local_model} ...")
            self._embedding_model = SentenceTransformer(
                self.config.embedding_local_model,
                device="cpu",
            )

        self.logger.info(f"  本地 BGE 编码: {len(texts)} 条 ...")
        embeddings = self._embedding_model.encode(
            texts,
            batch_size=8,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def _load_embedding_model(self):
        """延迟加载本地 Embedding 模型（兼容旧接口）"""
        if self._embedding_model is not None:
            return self._embedding_model

        try:
            from sentence_transformers import SentenceTransformer

            self.logger.info(f"加载 Embedding 模型: {self.config.embedding_local_model} ...")
            self._embedding_model = SentenceTransformer(
                self.config.embedding_local_model,
                device="cpu",
            )
            self.logger.info(f"Embedding 模型加载完成 (dim={self._embedding_model.get_sentence_embedding_dimension()})")
            return self._embedding_model

        except ImportError:
            self.logger.warning(
                "sentence-transformers 未安装。"
                "安装: pip install -e '.[embed]'"
            )
            return None

    def _get_chroma_collection(self):
        """获取或创建 ChromaDB collection"""
        if self._collection is not None:
            return self._collection

        try:
            import chromadb
            from chromadb.config import Settings

            self._chroma_client = chromadb.PersistentClient(
                path=self.config.chroma_persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )

            # 获取或创建 collection
            self._collection = self._chroma_client.get_or_create_collection(
                name=self.config.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            self.logger.info(
                f"ChromaDB collection 就绪: {self._collection.count()} 条记录"
            )
            return self._collection

        except ImportError:
            self.logger.warning(
                "chromadb 未安装。安装: pip install -e '.[rag]'"
            )
            return None

    def _embed_and_store(self, chunks: List[str], input_data: TranscriberOutput) -> None:
        """
        生成 Embedding 并存入 ChromaDB

        优先 DashScope API，失败则回退本地 BGE
        """
        embeddings = self._get_embeddings(chunks)
        if embeddings is None:
            return

        collection = self._get_chroma_collection()
        if collection is None:
            return

        # 存入 ChromaDB
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {
                "episode_id": input_data.episode_id or "unknown",
                "chunk_index": i,
                "word_count": len(chunk.split()),
            }
            for i, chunk in enumerate(chunks)
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        self.logger.info(f"  ChromaDB 存储完成: {len(chunks)} 条")

    # ================================================================
    # 混合检索 + Cross-encoder 重排序
    # ================================================================

    def _build_query(self, input_data: TranscriberOutput) -> str:
        """
        构建检索查询

        使用转录文本的关键开头作为查询
        """
        # 取前200词作为查询
        words = input_data.full_text.split()[:200]
        return " ".join(words)

    def _retrieve_similar(
        self, query: str, top_k: int = 5
    ) -> List[RetrievedChunk]:
        """
        混合检索: 语义检索 + BM25关键词 → 融合去重 → 重排序(必须)

        设计原则:
          - 每种检索方式各取 top_k 条最相关结果
          - 将所有召回的文本块合并去重
          - 重排序是必须步骤，对合并后的候选集精排，最终取 top_k

        流程:
          1. 语义检索: ChromaDB + DashScope Embedding → top_k
          2. BM25检索: 关键词稀疏检索 → top_k
          3. RRF 融合去重 (Reciprocal Rank Fusion)
          4. 重排序(必须): DashScope gte-rerank → 本地BM25回退 → top_k
        """
        # Step 1: 语义检索 — 取 top_k
        dense_results = self._chroma_retrieve(query, top_k)

        # Step 2: BM25 关键词检索 — 取 top_k
        sparse_results = self._bm25_retrieve(query, top_k)

        if not dense_results and not sparse_results:
            return []

        self.logger.info(
            f"  混合检索: 语义={len(dense_results)} 条 + BM25={len(sparse_results)} 条"
        )

        # Step 3: RRF 融合去重 — 合并两路检索的所有结果
        merged = self._rrf_fusion(dense_results, sparse_results, k=60)
        self.logger.info(f"  RRF 融合后: {len(merged)} 条候选")

        # Step 4: 重排序(必须) — 对全部候选集精排，取 top_k
        # 优先 API rerank，不可用时回退本地 BM25 重排序
        if self.config.rerank_available:
            try:
                merged = self._cross_encoder_rerank(query, merged, top_k)
                self.logger.info(f"  Cross-encoder 重排序完成 → top {len(merged)}")
            except Exception as e:
                self.logger.warning(f"  Cross-encoder 重排序失败: {e}，回退本地重排序")
                merged = self._local_rerank(query, merged, top_k)
        else:
            self.logger.info("  Rerank API 不可用，使用本地 BM25 重排序")
            merged = self._local_rerank(query, merged, top_k)

        return merged

    def _rrf_fusion(
        self,
        dense: List[RetrievedChunk],
        sparse: List[RetrievedChunk],
        k: int = 60,
    ) -> List[RetrievedChunk]:
        """
        Reciprocal Rank Fusion — 融合语义和关键词检索结果

        公式: RRF(d) = Σ 1/(k + rank_i(d))
        对每个文档在两路检索中的排名取倒数求和，排名越靠前得分越高
        """
        scores: dict = {}
        docs: dict = {}

        # 语义检索排名
        for rank, chunk in enumerate(dense, 1):
            key = chunk.chunk_id
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank)
            docs[key] = chunk

        # BM25 检索排名
        for rank, chunk in enumerate(sparse, 1):
            key = chunk.chunk_id
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank)
            if key not in docs:
                docs[key] = chunk

        # 按 RRF 分数排序
        sorted_ids = sorted(scores, key=scores.get, reverse=True)

        return [
            RetrievedChunk(
                chunk_id=doc_id,
                text=docs[doc_id].text,
                similarity_score=scores[doc_id],
                episode_id=docs[doc_id].episode_id,
                episode_title=docs[doc_id].episode_title,
                podcast_title=docs[doc_id].podcast_title,
            )
            for doc_id in sorted_ids
        ]

    def _cross_encoder_rerank(
        self, query: str, candidates: List[RetrievedChunk], top_n: int
    ) -> List[RetrievedChunk]:
        """
        Cross-encoder 重排序 — 通过 HTTP API 精排 (默认 DashScope qwen3-rerank)

        与双塔模型(Embedding)不同，Cross-encoder 同时输入 query+document，
        做全注意力计算，排序精度更高，但速度较慢，适合对候选集精排
        """
        import requests

        documents = [c.text for c in candidates]

        self.logger.info(f"  Cross-encoder 重排序: {len(candidates)} 条 → top {top_n}")

        resp = requests.post(
            self.config.rerank_base_url,
            headers={
                "Authorization": f"Bearer {self.config.rerank_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.rerank_model,
                "input": {
                    "query": query,
                    "documents": documents,
                },
                "parameters": {"top_n": top_n, "return_documents": False},
            },
            timeout=self.config.request_timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("code"):
            raise RuntimeError(f"Rerank API 返回错误: {data.get('code')} - {data.get('message')}")

        results = data.get("output", {}).get("results", [])

        # 按重排序结果重建
        reranked = []
        for item in results:
            idx = item["index"]
            original = candidates[idx]
            reranked.append(RetrievedChunk(
                chunk_id=original.chunk_id,
                text=original.text,
                similarity_score=item.get("relevance_score", original.similarity_score),
                episode_id=original.episode_id,
                episode_title=original.episode_title,
                podcast_title=original.podcast_title,
            ))

        return reranked

    def _local_rerank(
        self, query: str, candidates: List[RetrievedChunk], top_n: int
    ) -> List[RetrievedChunk]:
        """
        本地 BM25 重排序（Cross-encoder 不可用时的回退方案）

        使用 BM25 对候选集重新打分排序，确保重排序步骤始终执行。
        虽然精度不如 Cross-encoder，但比直接截断 RRF 结果更可靠，
        因为 BM25 会基于 query 重新计算每个候选的相关性，而非仅依赖排名融合。
        """
        from rank_bm25 import BM25Okapi

        if len(candidates) <= top_n:
            return candidates

        # 用 BM25 对候选集重新打分
        corpus = [c.text.lower().split() for c in candidates]
        tokenized_query = query.lower().split()

        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(tokenized_query)

        # 按 BM25 分数重新排序
        max_score = max(scores) if max(scores) > 0 else 1.0
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        self.logger.info(f"  本地 BM25 重排序: {len(candidates)} 条 → top {top_n}")

        return [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                text=c.text,
                similarity_score=min(score / max_score, 1.0),
                episode_id=c.episode_id,
                episode_title=c.episode_title,
                podcast_title=c.podcast_title,
            )
            for c, score in ranked[:top_n]
        ]

    def _chroma_retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """ChromaDB 语义检索"""
        collection = self._get_chroma_collection()
        if collection is None:
            return []

        # 生成查询向量
        query_embeddings = self._get_embeddings([query])
        if query_embeddings is None:
            return []

        # 检索
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=min(top_k, collection.count()),
        )

        retrieved = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                chunk = RetrievedChunk(
                    chunk_id=doc_id,
                    text=results["documents"][0][i] if results.get("documents") else "",
                    similarity_score=(
                        1.0 - results["distances"][0][i]
                        if results.get("distances")
                        else 0.0
                    ),
                )
                # 补充元数据
                if results.get("metadatas") and results["metadatas"][0]:
                    meta = results["metadatas"][0][i]
                    chunk.episode_id = meta.get("episode_id", "")

                retrieved.append(chunk)

        return retrieved

    def _bm25_retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """
        BM25 关键词检索（稀疏检索）

        使用 rank-bm25 库实现标准 BM25 算法，
        比简单关键词重叠度更精确，考虑了词频饱和和文档长度归一化
        """
        from rank_bm25 import BM25Okapi

        # 从 ChromaDB 获取所有文档
        all_docs = []
        try:
            collection = self._get_chroma_collection()
            if collection and collection.count() > 0:
                all_data = collection.get()
                all_docs = list(zip(
                    all_data["ids"],
                    all_data.get("documents", [""] * len(all_data["ids"])),
                ))
        except Exception:
            pass

        if not all_docs:
            return []

        # 构建 BM25 索引
        doc_ids = [doc_id for doc_id, _ in all_docs]
        corpus = [doc_text.lower().split() for _, doc_text in all_docs]
        tokenized_query = query.lower().split()

        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(tokenized_query)

        # 按 BM25 分数排序
        ranked = sorted(
            zip(doc_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # 归一化分数到 0-1
        max_score = ranked[0][1] if ranked else 1.0

        return [
            RetrievedChunk(
                chunk_id=doc_id,
                text=all_docs[i][1][:500],
                similarity_score=min(score / max(max_score, 0.001), 1.0),
            )
            for i, (doc_id, score) in enumerate(ranked[:top_k])
            if score > 0
        ]

    # ================================================================
    # 构建增强上下文
    # ================================================================

    def _build_augmented_context(
        self, current_text: str, retrieved: List[RetrievedChunk]
    ) -> str:
        """
        构建增强上下文

        格式: [检索到的历史相关段落] + [当前文本]
        """
        if not retrieved:
            return current_text

        parts = ["=== 历史相关段落 (RAG检索) ===\n"]

        for i, chunk in enumerate(retrieved, 1):
            parts.append(
                f"[段落 {i}] (相似度: {chunk.similarity_score:.3f})\n"
                f"{chunk.text}\n"
            )

        parts.append("\n=== 当前分析文本 ===\n")
        parts.append(current_text)

        return "\n".join(parts)