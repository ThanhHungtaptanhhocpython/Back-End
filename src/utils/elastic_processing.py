"""Elasticsearch Processing Utility.

This module provides the `ElasticProcessor` class to manage indices, 
bulk insert data, and execute robust text search queries for OCR and ASR data.
"""

import logging
from typing import Any, List, Dict

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from src.config.settings import get_settings

logger = logging.getLogger(__name__)

class ElasticProcessor:
    """Wrapper around the official Elasticsearch Python client."""

    def __init__(self, es_url: str | None = None) -> None:
        """Initialize the Elasticsearch client.
        
        Args:
            es_url: Optional connection URL. Defaults to the one in settings.
        """
        if es_url is None:
            settings = get_settings()
            es_url = settings.elasticsearch_url
            
        # For local development with xpack.security disabled, we don't need credentials.
        self.es = Elasticsearch([es_url])

    def create_indices(self, mappings_dict: Dict[str, Any]) -> None:
        """Create indices based on a mappings dictionary if they don't exist.
        
        Args:
            mappings_dict: Dictionary where keys are index names and values are 
                the settings/mappings configuration.
        """
        for index_name, body in mappings_dict.items():
            if not self.es.indices.exists(index=index_name):
                self.es.indices.create(index=index_name, body=body)
                logger.info(f"Created index '{index_name}' successfully.")
            else:
                logger.info(f"Index '{index_name}' already exists. Skipping creation.")

    def bulk_index_ocr(self, documents: List[Dict[str, Any]]) -> int:
        """Bulk index OCR documents into the `aic_ocr` index.
        
        Args:
            documents: List of OCR metadata dictionaries.
            
        Returns:
            The number of successfully indexed documents.
        """
        actions = [
            {
                "_index": "aic_ocr",
                "_source": doc
            }
            for doc in documents
        ]
        success, _ = bulk(self.es, actions, refresh=True)
        logger.info(f"Successfully bulk indexed {success} documents into 'aic_ocr'.")
        return success

    def bulk_index_asr(self, documents: List[Dict[str, Any]]) -> int:
        """Bulk index ASR documents into the `aic_asr` index.
        
        Args:
            documents: List of ASR transcript dictionaries.
            
        Returns:
            The number of successfully indexed documents.
        """
        actions = [
            {
                "_index": "aic_asr",
                "_source": doc
            }
            for doc in documents
        ]
        success, _ = bulk(self.es, actions, refresh=True)
        logger.info(f"Successfully bulk indexed {success} documents into 'aic_asr'.")
        return success

    def search_ocr(self, query: str, topk: int = 100) -> List[Dict[str, Any]]:
        """Search the `aic_ocr` index for matching text.
        
        Uses a boolean query that prioritizes exact phrase matches but falls
        back to multi_match for robust partial matching.
        
        Args:
            query: The search text.
            topk: Maximum number of results to return.
            
        Returns:
            A list of source documents with an added `_score` field.
        """
        body = {
            "query": {
                "bool": {
                    "should": [
                        {"match_phrase": {"ocr_text": {"query": query, "boost": 2.0}}},
                        {"multi_match": {"query": query, "fields": ["ocr_text"]}}
                    ]
                }
            },
            "size": topk
        }
        
        response = self.es.search(index="aic_ocr", body=body)
        
        results = []
        for hit in response["hits"]["hits"]:
            doc = hit["_source"]
            doc["_score"] = hit["_score"]
            results.append(doc)
            
        return results

    def search_asr(self, query: str, topk: int = 100) -> List[Dict[str, Any]]:
        """Search the `aic_asr` index for matching text.
        
        Args:
            query: The search text.
            topk: Maximum number of results to return.
            
        Returns:
            A list of source documents with an added `_score` field.
        """
        body = {
            "query": {
                "bool": {
                    "should": [
                        {"match_phrase": {"text": {"query": query, "boost": 2.0}}},
                        {"multi_match": {"query": query, "fields": ["text"]}}
                    ]
                }
            },
            "size": topk
        }
        
        response = self.es.search(index="aic_asr", body=body)
        
        results = []
        for hit in response["hits"]["hits"]:
            doc = hit["_source"]
            doc["_score"] = hit["_score"]
            results.append(doc)
            
        return results
