"""지원프로그램 의미 유사도 backend와 안전한 fallback 구현."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

import numpy as np

from src.utils import load_app_config


class SimilarityBackend(ABC):
    """query와 여러 문서의 0~1 유사도를 반환하는 인터페이스."""

    backend_name = "base"

    @abstractmethod
    def similarities(self, query: str, documents: Sequence[str]) -> np.ndarray:
        """문서 순서와 같은 길이의 유사도 배열을 반환한다."""


class TokenOverlapSimilarityBackend(SimilarityBackend):
    """외부 모델 없이 동작하는 최종 deterministic fallback."""

    backend_name = "token_overlap"

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[가-힣A-Za-z0-9]+", str(text).lower())
            if len(token) >= 2
        }

    def similarities(self, query: str, documents: Sequence[str]) -> np.ndarray:
        """토큰 Jaccard 유사도를 계산한다."""

        query_tokens = self._tokens(query)
        scores = []
        for document in documents:
            document_tokens = self._tokens(document)
            union = query_tokens | document_tokens
            scores.append(
                len(query_tokens & document_tokens) / len(union) if union else 0.0
            )
        return np.asarray(scores, dtype=float)


class TfidfSimilarityBackend(SimilarityBackend):
    """scikit-learn TF-IDF cosine 유사도 fallback."""

    backend_name = "tfidf"

    def similarities(self, query: str, documents: Sequence[str]) -> np.ndarray:
        """문자 n-gram 기반 TF-IDF로 한국어 띄어쓰기 변형에 대응한다."""

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        if not documents:
            return np.asarray([], dtype=float)
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        matrix = vectorizer.fit_transform([query, *documents])
        return cosine_similarity(matrix[0:1], matrix[1:]).ravel()


class SentenceTransformerSimilarityBackend(SimilarityBackend):
    """다국어 Sentence-Transformers 임베딩 기반 의미 유사도."""

    backend_name = "sentence_transformers"

    def __init__(self, model_name: str, local_files_only: bool = True) -> None:
        self.model_name = model_name
        self.local_files_only = local_files_only
        self._model: Any = None
        self._embedding_cache: dict[str, np.ndarray] = {}

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name, local_files_only=self.local_files_only
            )
        return self._model

    def similarities(self, query: str, documents: Sequence[str]) -> np.ndarray:
        """문서 임베딩을 재사용하며 query와 cosine 유사도를 계산한다."""

        if not documents:
            return np.asarray([], dtype=float)
        texts = [str(query), *(str(document) for document in documents)]
        missing_texts = list(
            dict.fromkeys(
                text for text in texts if text not in self._embedding_cache
            )
        )
        if missing_texts:
            encoded = self._load_model().encode(
                missing_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for text, embedding in zip(missing_texts, encoded):
                self._embedding_cache[text] = np.asarray(embedding, dtype=float)
        query_embedding = self._embedding_cache[texts[0]]
        document_embeddings = np.vstack(
            [self._embedding_cache[text] for text in texts[1:]]
        )
        return np.clip(document_embeddings @ query_embedding, 0.0, 1.0)


class ResilientSimilarityBackend(SimilarityBackend):
    """우선 backend 장애 시 순차적으로 fallback하는 wrapper."""

    backend_name = "resilient"

    def __init__(self, backends: Sequence[SimilarityBackend]) -> None:
        if not backends:
            raise ValueError("최소 하나의 similarity backend가 필요합니다.")
        self.backends = list(backends)
        self.last_backend_name = self.backends[-1].backend_name
        self.last_warning: str | None = None

    def similarities(self, query: str, documents: Sequence[str]) -> np.ndarray:
        """사용 가능한 첫 backend 결과를 반환한다."""

        failed_names = []
        for backend in self.backends:
            try:
                scores = backend.similarities(query, documents)
                if len(scores) != len(documents):
                    raise ValueError("similarity 결과 길이가 문서 수와 다릅니다.")
                self.last_backend_name = backend.backend_name
                self.last_warning = (
                    f"{', '.join(failed_names)} backend를 사용할 수 없어 "
                    f"{backend.backend_name} 방식으로 전환했습니다."
                    if failed_names
                    else None
                )
                return np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
            except Exception:
                failed_names.append(backend.backend_name)
        raise RuntimeError("사용 가능한 similarity backend가 없습니다.")


def create_similarity_backend(
    config: Mapping[str, Any] | None = None,
) -> ResilientSimilarityBackend:
    """설정에 따라 Sentence-Transformers 우선 fallback 체인을 만든다."""

    active_config = config or load_app_config()
    similarity_config = active_config["recommendation"]["similarity"]
    backends: list[SimilarityBackend] = []
    if bool(similarity_config["prefer_sentence_transformers"]):
        backends.append(
            SentenceTransformerSimilarityBackend(
                model_name=str(similarity_config["sentence_transformer_model"]),
                local_files_only=bool(similarity_config["local_files_only"]),
            )
        )
    backends.extend([TfidfSimilarityBackend(), TokenOverlapSimilarityBackend()])
    return ResilientSimilarityBackend(backends)
