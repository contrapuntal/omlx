# SPDX-License-Identifier: Apache-2.0
"""
Embedding engine for oMLX.

This module provides an engine for generating text embeddings using
mlx-embeddings. Unlike LLM engines, embedding engines don't support
streaming or chat completion.
"""

import asyncio
import gc
import logging
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx

from ..engine_core import get_mlx_executor
from ..models.embedding import EmbeddingOutput, MLXEmbeddingModel
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)

_DEFAULT_MAX_MASK_ELEMENTS = 2_000_000_000


def _plan_length_bucketed_batches(
    lengths: List[int],
    max_items: int,
    max_mask_elements: int,
) -> List[List[int]]:
    """Group input indices into batches that minimize padding and bound mask size.

    Inputs are sorted by length so each batch pads only to a length close to its
    own members (cutting wasted compute), and a batch is closed when adding the
    next item would exceed ``max_items`` or push
    ``len(batch) * max_len_in_batch**2`` over ``max_mask_elements`` — the size of
    the dense attention mask, which is what OOMs Metal when a long input pads a
    full batch up to the model's context. A single item always gets its own batch
    even if it alone exceeds the element budget (a lone input has no padding, so
    it builds no dense mask).

    Args:
        lengths: Per-input (post-truncation) token length, indexed by input.
        max_items: Hard cap on items per batch.
        max_mask_elements: Cap on ``items * max_len**2`` per batch; ``<= 0``
            disables the mask cap (item count only).

    Returns:
        Batches, each a list of original input indices. Every index appears once.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches: List[List[int]] = []
    current: List[int] = []
    current_max = 0
    for idx in order:
        length = max(1, lengths[idx])
        if current:
            new_max = length if length > current_max else current_max
            over_items = len(current) + 1 > max_items
            over_mask = (
                max_mask_elements > 0
                and (len(current) + 1) * new_max * new_max > max_mask_elements
            )
            if over_items or over_mask:
                batches.append(current)
                current = []
                current_max = 0
        current.append(idx)
        if length > current_max:
            current_max = length
    if current:
        batches.append(current)
    return batches


def _input_length(item: Union[str, Dict[str, str]]) -> int:
    """Approximate token cost of one embedding input, used only to group similar sizes."""
    if isinstance(item, str):
        return len(item)
    if isinstance(item, dict):
        return sum(len(v) for v in item.values() if isinstance(v, str))
    return 0


class EmbeddingEngine(BaseNonStreamingEngine):
    """
    Engine for generating text embeddings.

    This engine wraps MLXEmbeddingModel and provides async methods
    for integration with the oMLX server.

    Unlike BaseEngine, this doesn't support streaming or chat
    since embeddings are computed in a single forward pass.
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = False,
        batch_size: int | None = None,
        *,
        scheduler_config: Any | None = None,
    ):
        """
        Initialize the embedding engine.

        Args:
            model_name: HuggingFace model name or local path
            trust_remote_code: Allow loaders to execute custom Python shipped
                with the model repo. Off by default for security (issue #926).
            batch_size: Explicit per-forward input chunk size override.
            scheduler_config: Shared scheduler configuration. Embedding uses
                embedding_batch_size as its per-forward input chunk size.
        """
        super().__init__()
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        if batch_size is None:
            batch_size = (
                getattr(scheduler_config, "embedding_batch_size", 32)
                if scheduler_config is not None
                else 32
            )
        self._batch_size = max(1, int(batch_size))
        self._max_mask_elements = max(
            0,
            int(
                getattr(
                    scheduler_config,
                    "embedding_max_mask_elements",
                    _DEFAULT_MAX_MASK_ELEMENTS,
                )
                if scheduler_config is not None
                else _DEFAULT_MAX_MASK_ELEMENTS
            ),
        )
        self._model: Optional[MLXEmbeddingModel] = None

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def processor(self) -> Any:
        """Get the processor/tokenizer."""
        return self._model.processor if self._model else None

    @property
    def hidden_size(self) -> Optional[int]:
        """Get the embedding dimension."""
        return self._model.hidden_size if self._model else None

    async def start(self) -> None:
        """Start the engine (load model if not loaded).

        Model loading runs on the global MLX executor to avoid Metal
        command buffer races with concurrent BatchGenerator steps.
        """
        if self._model is not None:
            return

        logger.info(f"Starting embedding engine: {self._model_name}")
        self._model = MLXEmbeddingModel(
            self._model_name, trust_remote_code=self._trust_remote_code
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(get_mlx_executor(), self._model.load)
        logger.info(f"Embedding engine started: {self._model_name}")

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        if self._model is None:
            return

        logger.info(f"Stopping embedding engine: {self._model_name}")
        model = self._model
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(get_mlx_executor(), model.close)
        self._model = None

        gc.collect()
        logger.info(f"Embedding engine stopped: {self._model_name}")

    async def embed(
        self,
        texts: Union[List[str], List[Dict[str, str]]],
        max_length: int | None = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> EmbeddingOutput:
        """
        Generate embeddings for input texts.

        Args:
            texts: List of input texts
            max_length: Maximum token length for each text. If omitted, the
                model resolves its configured limit.
            padding: Whether to pad shorter sequences
            truncation: Whether to truncate longer sequences

        Returns:
            EmbeddingOutput with embeddings and token count
        """
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")

        model = self._model
        input_items = [texts] if isinstance(texts, str) else list(texts)

        if not input_items:
            return EmbeddingOutput(embeddings=[], total_tokens=0, dimensions=0)

        batch_size = self._batch_size
        activity_id = self._begin_activity(
            "embedding",
            detail="Embedding",
            total_items=len(input_items),
            metadata={"input_count": len(input_items), "batch_size": batch_size},
        )
        try:
            loop = asyncio.get_running_loop()
            n = len(input_items)

            # Measure token lengths so similar-length inputs batch together
            # (less padding => faster) and a long input cannot pad a full batch
            # up to the model's context and OOM the dense attention mask.
            # Tokenization is CPU-only; run it on the MLX executor to stay
            # serialized with the forward passes.
            def _measure():
                return model.measure_lengths(
                    input_items, max_length=max_length, truncation=truncation
                )

            try:
                lengths = await loop.run_in_executor(get_mlx_executor(), _measure)
            except Exception as exc:  # never let measurement break embedding
                logger.warning(
                    "embedding length measurement failed (%s); "
                    "falling back to fixed-size batches",
                    exc,
                )
                lengths = None

            if isinstance(lengths, list) and len(lengths) == n:
                planned = _plan_length_bucketed_batches(
                    lengths, batch_size, self._max_mask_elements
                )
            else:
                # Measurement unavailable: fall back to the character-length
                # proxy for ordering only, which still strips most of the
                # padding. Its units are characters, not tokens, so it cannot be
                # trusted against the token-based mask budget — group by size
                # under the item cap alone.
                planned = _plan_length_bucketed_batches(
                    [_input_length(item) for item in input_items], batch_size, 0
                )

            results: List[Optional[List[float]]] = [None] * n
            total_tokens = 0
            dimensions = 0
            completed = 0

            for batch_indices in planned:
                batch = [input_items[i] for i in batch_indices]

                def _embed_sync(batch=batch):
                    try:
                        return model.embed(
                            inputs=batch,
                            max_length=max_length,
                            padding=padding,
                            truncation=truncation,
                        )
                    finally:
                        mx.synchronize()
                        mx.clear_cache()

                output = await loop.run_in_executor(get_mlx_executor(), _embed_sync)
                if len(output.embeddings) != len(batch_indices):
                    raise RuntimeError(
                        "embedding model returned "
                        f"{len(output.embeddings)} vectors for a batch of "
                        f"{len(batch_indices)} inputs"
                    )
                for i, emb in zip(batch_indices, output.embeddings):
                    results[i] = emb
                total_tokens += output.total_tokens
                if output.dimensions:
                    dimensions = output.dimensions
                completed += len(batch_indices)
                self._update_activity(
                    activity_id,
                    completed_items=completed,
                    token_count=total_tokens,
                    dimensions=dimensions,
                )

            if any(emb is None for emb in results):
                raise RuntimeError(
                    "embedding engine produced fewer vectors than inputs"
                )
            embeddings: List[List[float]] = [emb for emb in results if emb is not None]
            output = EmbeddingOutput(
                embeddings=embeddings,
                total_tokens=total_tokens,
                dimensions=dimensions,
            )
            self._update_activity(
                activity_id,
                token_count=output.total_tokens,
                dimensions=output.dimensions,
            )
            return output
        finally:
            self._end_activity(activity_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
            "hidden_size": self.hidden_size,
            "batch_size": self._batch_size,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self._model is None:
            return {"loaded": False, "model_name": self._model_name}
        return self._model.get_model_info()

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<EmbeddingEngine model={self._model_name} status={status}>"
