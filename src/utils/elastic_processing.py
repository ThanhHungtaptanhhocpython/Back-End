"""Elasticsearch Processing Utility.

This module provides the `ElasticProcessor` class to manage indices,
bulk insert data, and execute robust text search queries for OCR and ASR data.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterable
from typing import Any, Dict, List

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


DEFAULT_BULK_CHUNK_SIZE = 250
DEFAULT_BULK_MAX_CHUNK_BYTES = 10 * 1024 * 1024
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 5
RETRYABLE_BULK_STATUS = (429, 500, 502, 503, 504)


class ElasticProcessor:
    """Wrapper around the official Elasticsearch Python client."""

    def __init__(
        self,
        es_url: str | None = None,
        *,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """Initialize the Elasticsearch client.

        Args:
            es_url: Optional connection URL. Defaults to the one in settings.
            request_timeout: Per-request timeout in seconds for search/bulk calls.
            max_retries: Transport-level retry count for transient failures.
        """
        if es_url is None:
            settings = get_settings()
            es_url = settings.elasticsearch_url

        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.es = Elasticsearch(
            [es_url],
            request_timeout=request_timeout,
            retry_on_timeout=True,
            max_retries=max_retries,
            retry_on_status=RETRYABLE_BULK_STATUS,
        )

    def create_indices(self, mappings_dict: Dict[str, Any]) -> None:
        """Create indices based on a mappings dictionary if they don't exist.

        Args:
            mappings_dict: Dictionary where keys are index names and values are
                the settings/mappings configuration.
        """
        for index_name, body in mappings_dict.items():
            if not self.es.indices.exists(index=index_name):
                self.es.indices.create(index=index_name, body=body)
                logger.info("Created index '%s' successfully.", index_name)
            else:
                logger.info("Index '%s' already exists. Skipping creation.", index_name)

    @staticmethod
    def _ocr_doc_id(doc: Dict[str, Any]) -> str:
        """Stable OCR id so rerunning an import overwrites instead of duplicating."""
        key = "|".join(
            str(doc.get(field, ""))
            for field in (
                "vector_id",
                "faiss_id",
                "video_id",
                "frame_name",
                "timestamp",
                "ocr_source_timestamp",
                "ocr_text",
            )
        )
        return hashlib.sha1(key.encode("utf-8")).hexdigest()
    @staticmethod
    def _asr_doc_id(doc: Dict[str, Any]) -> str:
        key = "|".join(
            str(doc.get(field, ""))
            for field in (
                "video_id",
                "start_time",
                "end_time",
                "nearest_vector_id",
                "nearest_faiss_id",
                "text",
            )
        )
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    def _bulk_index_documents(
        self,
        documents: Iterable[Dict[str, Any]],
        *,
        index: str,
        doc_id: Callable[[Dict[str, Any]], str],
        chunk_size: int = DEFAULT_BULK_CHUNK_SIZE,
        request_timeout: float | None = None,
        max_retries: int | None = None,
        refresh: bool = True,
    ) -> int:
        """Index documents with small chunks, retry, and one optional final refresh."""
        timeout = self.request_timeout if request_timeout is None else request_timeout
        retries = self.max_retries if max_retries is None else max_retries

        def actions():
            for doc in documents:
                yield {
                    "_op_type": "index",
                    "_index": index,
                    "_id": doc_id(doc),
                    "_source": doc,
                }

        success = 0
        errors: list[Dict[str, Any]] = []
        for ok, item in streaming_bulk(
            self.es,
            actions(),
            chunk_size=chunk_size,
            max_chunk_bytes=DEFAULT_BULK_MAX_CHUNK_BYTES,
            raise_on_error=False,
            raise_on_exception=False,
            max_retries=retries,
            initial_backoff=2,
            max_backoff=60,
            retry_on_status=RETRYABLE_BULK_STATUS,
            request_timeout=timeout,
        ):
            if ok:
                success += 1
                if success % 10_000 == 0:
                    logger.info("Indexed %s documents into '%s'.", success, index)
                continue
            errors.append(item)
            if len(errors) <= 3:
                logger.warning("Bulk index failure in '%s': %s", index, item)

        if refresh:
            self.es.options(request_timeout=timeout).indices.refresh(index=index)
        if errors:
            raise RuntimeError(
                f"Failed to index {len(errors)} documents into '{index}'. First error: {errors[0]}"
            )
        logger.info("Successfully bulk indexed %s documents into '%s'.", success, index)
        return success

    def bulk_index_ocr(
        self,
        documents: List[Dict[str, Any]],
        *,
        index: str = "aic_ocr",
        chunk_size: int = DEFAULT_BULK_CHUNK_SIZE,
        request_timeout: float | None = None,
        max_retries: int | None = None,
        refresh: bool = True,
    ) -> int:
        """Bulk index OCR documents into the configured OCR index."""
        return self._bulk_index_documents(
            documents,
            index=index,
            doc_id=self._ocr_doc_id,
            chunk_size=chunk_size,
            request_timeout=request_timeout,
            max_retries=max_retries,
            refresh=refresh,
        )

    def bulk_index_asr(
        self,
        documents: List[Dict[str, Any]],
        *,
        index: str = "aic_asr",
        chunk_size: int = DEFAULT_BULK_CHUNK_SIZE,
        request_timeout: float | None = None,
        max_retries: int | None = None,
        refresh: bool = True,
    ) -> int:
        """Bulk index ASR documents into the configured ASR index."""
        return self._bulk_index_documents(
            documents,
            index=index,
            doc_id=self._asr_doc_id,
            chunk_size=chunk_size,
            request_timeout=request_timeout,
            max_retries=max_retries,
            refresh=refresh,
        )

    def search_ocr(self, query: str, topk: int = 100) -> List[Dict[str, Any]]:
        """Search the `aic_ocr` index for matching text.

        Uses a boolean query that prioritizes exact phrase matches but falls
        back to multi_match for robust partial matching.
        """
        body = {
            "query": {
                "bool": {
                    "should": [
                        {"match_phrase": {"ocr_text": {"query": query, "boost": 2.0}}},
                        {"multi_match": {"query": query, "fields": ["ocr_text"]}},
                    ]
                }
            },
            "size": topk,
        }

        response = self.es.options(request_timeout=self.request_timeout).search(index="aic_ocr", body=body)

        results = []
        for hit in response["hits"]["hits"]:
            doc = hit["_source"]
            doc["_score"] = hit["_score"]
            results.append(doc)

        return results

    def search_asr(self, query: str, topk: int = 100) -> List[Dict[str, Any]]:
        """Search the `aic_asr` index for matching text."""
        body = {
            "query": {
                "bool": {
                    "should": [
                        {"match_phrase": {"text": {"query": query, "boost": 2.0}}},
                        {"multi_match": {"query": query, "fields": ["text"]}},
                    ]
                }
            },
            "size": topk,
        }

        response = self.es.options(request_timeout=self.request_timeout).search(index="aic_asr", body=body)

        results = []
        for hit in response["hits"]["hits"]:
            doc = hit["_source"]
            doc["_score"] = hit["_score"]
            results.append(doc)

        return results