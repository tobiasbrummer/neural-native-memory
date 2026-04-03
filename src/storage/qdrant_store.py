"""Qdrant storage backend for token-level retrieval/injection vectors."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


def _import_qdrant():
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
    except ImportError as exc:
        raise RuntimeError(
            "qdrant-client is not installed. Install with: uv pip install qdrant-client"
        ) from exc
    return QdrantClient, models


@dataclass(frozen=True)
class TokenVectorRecord:
    point_id: str
    entry_id: str
    entity_id: str
    token_index: int
    token_id: int
    retrieval_vector: Sequence[float]
    injection_vector: Sequence[float]
    model_id: str
    retrieval_layer: int
    injection_layer: int
    timestamp_created: int
    supersedes: Optional[str] = None
    fsrs_score: Optional[float] = None
    source_id: Optional[str] = None
    source_title: Optional[str] = None


class NNMQdrantTokenStore:
    """Token-level store with named vectors: retrieval + injection."""

    def __init__(
        self,
        *,
        url: Optional[str] = "http://localhost:6333",
        path: Optional[str] = None,
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
        timeout: float = 120.0,
        check_compatibility: bool = True,
    ) -> None:
        QdrantClient, _ = _import_qdrant()
        if path:
            self.client = QdrantClient(path=path, timeout=timeout, check_compatibility=check_compatibility)
        else:
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                prefer_grpc=prefer_grpc,
                timeout=timeout,
                check_compatibility=check_compatibility,
            )

    def ensure_collection(
        self,
        *,
        collection_name: str,
        vector_size: int,
        recreate: bool = False,
        on_disk: bool = True,
        use_scalar_int8: bool = True,
    ) -> None:
        _, models = _import_qdrant()
        vectors_config = {
            "retrieval": models.VectorParams(
                size=int(vector_size),
                distance=models.Distance.COSINE,
                on_disk=bool(on_disk),
            ),
            "injection": models.VectorParams(
                size=int(vector_size),
                distance=models.Distance.COSINE,
                on_disk=bool(on_disk),
            ),
        }

        quantization_config = None
        if use_scalar_int8:
            quantization_config = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            )

        if recreate and self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name=collection_name)

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                quantization_config=quantization_config,
            )
        self._ensure_payload_indexes(collection_name)

    def collection_exists(self, collection_name: str) -> bool:
        return bool(self.client.collection_exists(collection_name))

    def list_collections(self) -> List[str]:
        resp = self.client.get_collections()
        out: List[str] = []
        collections = getattr(resp, "collections", None)
        if collections is not None:
            for item in collections:
                name = getattr(item, "name", None)
                if name:
                    out.append(str(name))
            return sorted(set(out))

        if isinstance(resp, dict):
            block = resp.get("result", resp)
            rows = block.get("collections", []) if isinstance(block, dict) else []
            for item in rows:
                if isinstance(item, dict) and "name" in item:
                    out.append(str(item["name"]))
        return sorted(set(out))

    def _ensure_payload_indexes(self, collection_name: str) -> None:
        _, models = _import_qdrant()
        index_fields = [
            ("model_id", models.PayloadSchemaType.KEYWORD),
            ("entry_id", models.PayloadSchemaType.KEYWORD),
            ("entity_id", models.PayloadSchemaType.KEYWORD),
            ("token_index", models.PayloadSchemaType.INTEGER),
            ("token_id", models.PayloadSchemaType.INTEGER),
            ("retrieval_layer", models.PayloadSchemaType.INTEGER),
            ("injection_layer", models.PayloadSchemaType.INTEGER),
            ("timestamp_created", models.PayloadSchemaType.INTEGER),
        ]
        for field_name, schema in index_fields:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=schema,
                    wait=True,
                )
            except Exception:
                # Local mode / older servers may report already exists or unsupported.
                pass

    def upsert_tokens(
        self,
        *,
        collection_name: str,
        records: Iterable[TokenVectorRecord],
        wait: bool = True,
        batch_size: int = 256,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ) -> int:
        _, models = _import_qdrant()

        count = 0
        batch: List[object] = []
        for rec in records:
            payload: Dict[str, object] = {
                "entry_id": rec.entry_id,
                "entity_id": rec.entity_id,
                "token_index": int(rec.token_index),
                "token_id": int(rec.token_id),
                "model_id": rec.model_id,
                "retrieval_layer": int(rec.retrieval_layer),
                "injection_layer": int(rec.injection_layer),
                "timestamp_created": int(rec.timestamp_created),
            }
            if rec.supersedes:
                payload["supersedes"] = rec.supersedes
            if rec.fsrs_score is not None:
                payload["fsrs_score"] = float(rec.fsrs_score)
            if rec.source_id is not None:
                payload["source_id"] = rec.source_id
            if rec.source_title is not None:
                payload["source_title"] = rec.source_title

            point = models.PointStruct(
                id=rec.point_id,
                vector={
                    "retrieval": [float(x) for x in rec.retrieval_vector],
                    "injection": [float(x) for x in rec.injection_vector],
                },
                payload=payload,
            )
            batch.append(point)
            count += 1
            if len(batch) >= batch_size:
                self._upsert_batch(
                    collection_name=collection_name,
                    batch=batch,
                    wait=wait,
                    max_retries=max_retries,
                    retry_base_delay=retry_base_delay,
                )
                batch = []

        if batch:
            self._upsert_batch(
                collection_name=collection_name,
                batch=batch,
                wait=wait,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay,
            )
        return count

    def _upsert_batch(
        self,
        *,
        collection_name: str,
        batch: Sequence[object],
        wait: bool,
        max_retries: int,
        retry_base_delay: float,
    ) -> None:
        attempts = max(0, int(max_retries)) + 1
        for attempt in range(1, attempts + 1):
            try:
                self.client.upsert(collection_name=collection_name, points=list(batch), wait=wait)
                return
            except Exception as exc:
                if attempt >= attempts:
                    raise
                delay = float(retry_base_delay) * (2 ** (attempt - 1))
                logger.warning(
                    "Qdrant upsert failed (attempt %d/%d, batch=%d): %s. Retrying in %.1fs",
                    attempt,
                    attempts,
                    len(batch),
                    exc,
                    delay,
                )
                time.sleep(max(delay, 0.0))

    def search_retrieval(
        self,
        *,
        collection_name: str,
        query_vector: Sequence[float],
        limit: int = 10,
        model_id: Optional[str] = None,
        min_score: Optional[float] = None,
        exclude_token_ids: Optional[Sequence[int]] = None,
        with_vectors: bool = False,
        with_payload: bool = True,
    ) -> List[object]:
        _, models = _import_qdrant()
        must_conditions: List[object] = []
        must_not_conditions: List[object] = []
        if model_id:
            must_conditions.append(
                models.FieldCondition(
                    key="model_id",
                    match=models.MatchValue(value=model_id),
                )
            )

        if exclude_token_ids:
            seen: set[int] = set()
            for token_id in exclude_token_ids:
                tid = int(token_id)
                if tid in seen:
                    continue
                seen.add(tid)
                must_not_conditions.append(
                    models.FieldCondition(
                        key="token_id",
                        match=models.MatchValue(value=tid),
                    )
                )

        query_filter = None
        if must_conditions or must_not_conditions:
            query_filter = models.Filter(
                must=must_conditions or None,
                must_not=must_not_conditions or None,
            )
        named_query = models.NamedVector(name="retrieval", vector=[float(x) for x in query_vector])
        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=collection_name,
                query_vector=named_query,
                query_filter=query_filter,
                limit=int(limit),
                score_threshold=min_score,
                with_vectors=with_vectors,
                with_payload=with_payload,
            )
        response = self.client.query_points(
            collection_name=collection_name,
            query=[float(x) for x in query_vector],
            using="retrieval",
            query_filter=query_filter,
            limit=int(limit),
            score_threshold=min_score,
            with_vectors=with_vectors,
            with_payload=with_payload,
        )
        if isinstance(response, list):
            return response
        if hasattr(response, "points"):
            return list(response.points)
        return list(response)

    def fetch_entry_tokens(
        self,
        *,
        collection_name: str,
        entry_id: str,
        with_vectors: bool = True,
        page_size: int = 256,
    ) -> List[object]:
        _, models = _import_qdrant()
        flt = models.Filter(
            must=[models.FieldCondition(key="entry_id", match=models.MatchValue(value=entry_id))]
        )
        offset = None
        out: List[object] = []
        while True:
            points, offset = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=flt,
                with_payload=True,
                with_vectors=with_vectors,
                limit=int(page_size),
                offset=offset,
            )
            out.extend(points)
            if offset is None:
                break
        out.sort(key=lambda p: int((p.payload or {}).get("token_index", 0)))
        return out

    def entry_exists(
        self,
        *,
        collection_name: str,
        entry_id: str,
    ) -> bool:
        _, models = _import_qdrant()
        flt = models.Filter(
            must=[models.FieldCondition(key="entry_id", match=models.MatchValue(value=entry_id))]
        )
        points, _ = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=flt,
            with_payload=False,
            with_vectors=False,
            limit=1,
        )
        return bool(points)
