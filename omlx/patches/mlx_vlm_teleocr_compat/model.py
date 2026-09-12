"""TeleOCR decoder math with upstream Qwen2.5-VL vision and position handling."""

from dataclasses import dataclass, fields

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.base import scaled_dot_product_attention
from mlx_vlm.models.qwen2_5_vl.config import (
    ModelConfig as QwenModelConfig,
)
from mlx_vlm.models.qwen2_5_vl.config import (
    TextConfig as QwenTextConfig,
)
from mlx_vlm.models.qwen2_5_vl.config import (
    VisionConfig,
)
from mlx_vlm.models.qwen2_5_vl.language import (
    LanguageModel as QwenLanguageModel,
)
from mlx_vlm.models.qwen2_5_vl.language import (
    Qwen2RotaryEmbedding,
    apply_multimodal_rotary_pos_emb,
)
from mlx_vlm.models.qwen2_5_vl.qwen2_5_vl import Model as QwenModel
from mlx_vlm.models.qwen2_5_vl.vision import VisionModel as QwenVisionModel


@dataclass
class TextConfig(QwenTextConfig):
    head_dim: int | None = None
    use_sliding_window: bool = False
    layer_types: list[str] | None = None

    def __post_init__(self):
        if self.use_sliding_window or (
            self.layer_types is not None
            and (
                len(self.layer_types) != self.num_hidden_layers
                or any(kind != "full_attention" for kind in self.layer_types)
            )
        ):
            raise ValueError("TeleOCR supports full attention layers only")
        if self.rope_scaling:
            self.rope_scaling = dict(self.rope_scaling)
            self.rope_scaling.setdefault(
                "type", self.rope_scaling.get("rope_type", "default")
            )
        super().__post_init__()
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


@dataclass
class ModelConfig(QwenModelConfig):
    tie_word_embeddings: bool = False
    head_dim: int | None = None

    @classmethod
    def from_dict(cls, params):
        data = dict(params)
        text = {
            key: value
            for key, value in data.items()
            if key in {f.name for f in fields(TextConfig)}
        }
        text.update(data.get("text_config") or {})
        # TeleOCR's pinned Transformers runtime hoists the nested setting:
        # its original checkpoint has nested=True and no root setting, and
        # loads with the head sharing the embedding parameter.
        data["tie_word_embeddings"] = data.get(
            "tie_word_embeddings", text.get("tie_word_embeddings", True)
        )
        text["tie_word_embeddings"] = data["tie_word_embeddings"]
        text.setdefault("model_type", "qwen2_5_vl")
        data["text_config"] = TextConfig.from_dict(text)
        data["head_dim"] = data["text_config"].head_dim
        data["vision_config"] = VisionConfig.from_dict(data.get("vision_config") or {})
        return cls(
            **{k: v for k, v in data.items() if k in {f.name for f in fields(cls)}}
        )


class Attention(nn.Module):
    """Bias-free GQA with explicit head width and pre-rotary Q/K RMSNorm."""

    def __init__(self, args: TextConfig):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            args.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, args.hidden_size, bias=False
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=args.max_position_embeddings,
            base=args.rope_theta,
            rope_scaling=args.rope_scaling,
        )

    def __call__(
        self, x, mask=None, cache=None, position_ids=None, position_embeddings=None
    ):
        batch, length, _ = x.shape
        queries = self.q_norm(
            self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        )
        keys = self.k_norm(
            self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        )
        queries, keys = queries.transpose(0, 2, 1, 3), keys.transpose(0, 2, 1, 3)
        values = (
            self.v_proj(x)
            .reshape(batch, length, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        if position_embeddings is None:
            if position_ids is None:
                offset = cache.offset if cache is not None else 0
                position_ids = mx.arange(offset, offset + length)[None, :]
            queries, keys = self.rotary_emb.apply_rotary(
                queries, keys, position_ids, unsqueeze_dim=1
            )
        else:
            queries, keys = apply_multimodal_rotary_pos_emb(
                queries, keys, *position_embeddings
            )
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        if isinstance(mask, mx.array):
            mask = mask[..., : keys.shape[-2]]
        output = scaled_dot_product_attention(
            queries, keys, values, cache, scale=self.scale, mask=mask
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class LanguageModel(QwenLanguageModel):
    def __init__(self, args, config):
        super().__init__(args, config)
        for layer in self.model.layers:
            layer.self_attn = Attention(args)

    @property
    def head_dim(self):
        return self.args.head_dim


class VisionModel(QwenVisionModel):
    def sanitize(self, weights):
        # Shape-check against the configured convolution rather than upstream's
        # heuristic, which confuses RGB channels with small patch dimensions.
        cfg = self.config
        hf_shape = (
            cfg.hidden_size,
            cfg.in_channels,
            cfg.temporal_patch_size,
            cfg.patch_size,
            cfg.patch_size,
        )
        mlx_shape = (
            cfg.hidden_size,
            cfg.temporal_patch_size,
            cfg.patch_size,
            cfg.patch_size,
            cfg.in_channels,
        )
        result = {}
        for key, value in weights.items():
            if key.rsplit(".", 1)[-1] == "position_ids":
                continue
            if key.endswith("patch_embed.proj.weight"):
                if tuple(value.shape) == mlx_shape:
                    pass
                elif tuple(value.shape) == hf_shape:
                    value = value.transpose(0, 2, 3, 4, 1)
                else:
                    raise ValueError(
                        f"Unexpected TeleOCR patch embedding shape: {value.shape}"
                    )
            result[key] = value
        return result


class Model(QwenModel):
    omlx_adapter = "teleocr"

    def load_weights(self, weights, strict=True):
        # Calibration requests strict=False for other architectures' intentional
        # extra tensors. TeleOCR has no such exception: a marker is only a hint,
        # and missing norms/bias mismatches must never reach inference.
        return super().load_weights(weights, strict=True)

    def __init__(self, config):
        # mlx-vlm rebuilds nested configs after from_dict; restore normalized
        # root/nested values before the parent initializes any model state.
        config.text_config.tie_word_embeddings = config.tie_word_embeddings
        if config.head_dim is not None:
            config.text_config.head_dim = config.head_dim
        super().__init__(config)
        # The parent uses module-level component classes, not subclass factories.
        # Preserve its initialization while replacing the stock components.
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def sanitize(self, weights):
        """Map original HF and current HF names; already-MLX keys stay intact."""
        from .checkpoint import canonical_weight_key

        result = {}
        for key, value in weights.items():
            key = canonical_weight_key(key)
            if self.config.tie_word_embeddings and key.startswith(
                "language_model.lm_head."
            ):
                continue
            if key in result:
                raise ValueError(f"Duplicate TeleOCR tensor alias: {key}")
            result[key] = value
        return result
