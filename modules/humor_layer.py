"""Axis-aware transformer layers for 3D attention over mobility tensors.

This module provides a drop-in replacement for the old Attention3D stack,
but exposes attention by axis (sequence, feature, neighbor) directly.
Input/Output tensor contract:
    x: (batch_size, seq_len, num_feats, clique_size, feature_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AxisTransformerEncoderLayer(nn.Module):
	"""Transformer encoder layer applied along a chosen axis of an N-D tensor."""

	def __init__(
		self,
		d_model: int,
		nhead: int,
		dim_feedforward: int,
		dropout: float = 0.1,
		layer_norm_eps: float = 1e-5,
	) -> None:
		super().__init__()
		self.self_attn = nn.MultiheadAttention(
			embed_dim=d_model,
			num_heads=nhead,
			dropout=dropout,
			batch_first=True,
		)
		self.linear1 = nn.Linear(d_model, dim_feedforward)
		self.linear2 = nn.Linear(dim_feedforward, d_model)
		self.dropout = nn.Dropout(dropout)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)
		self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
		self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

	def forward(
		self,
		x: torch.Tensor,
		axis: int,
		attn_mask: torch.Tensor | None = None,
		key_padding_mask: torch.Tensor | None = None,
		need_weights: bool = False,
	) -> tuple[torch.Tensor, torch.Tensor | None]:
		attn_out, attn_weights = self._attend_along_axis(
			x,
			axis=axis,
			attn_mask=attn_mask,
			key_padding_mask=key_padding_mask,
			need_weights=need_weights,
		)
		x = self.norm1(x + self.dropout1(attn_out))
		ff = self.linear2(self.dropout(F.relu(self.linear1(x))))
		x = self.norm2(x + self.dropout2(ff))
		return x, attn_weights

	def _attend_along_axis(
		self,
		x: torch.Tensor,
		axis: int,
		attn_mask: torch.Tensor | None,
		key_padding_mask: torch.Tensor | None,
		need_weights: bool = False,
	) -> tuple[torch.Tensor, torch.Tensor | None]:
		# Attention axis is specified over non-embedding dimensions.
		n_axes = x.dim() - 1
		if axis < 0:
			axis += n_axes
		if axis < 0 or axis >= n_axes:
			raise ValueError(f"axis must be in [0, {n_axes - 1}], got {axis}")

		moved = x.movedim(axis, -2)  # (..., L, D)
		lead_shape = moved.shape[:-2]
		seq_len = moved.shape[-2]
		d_model = moved.shape[-1]

		flat = moved.reshape(-1, seq_len, d_model)

		if key_padding_mask is not None:
			kpm = key_padding_mask.reshape(-1, seq_len)
		else:
			kpm = None

		out, attn_weights = self.self_attn(
			flat,
			flat,
			flat,
			attn_mask=attn_mask,
			key_padding_mask=kpm,
			need_weights=need_weights,
			is_causal=False,
		)
		out = out.reshape(*lead_shape, seq_len, d_model)
		out = out.movedim(-2, axis)
		return out, attn_weights


class HumorLayer(nn.Module):
    """One 3D-attention block: sequence -> feature -> neighbor.

    forward() takes an optional ``attn_mask`` for the sequence-axis attention
    (e.g. a causal triangular mask used by ablation pretraining). Default
    ``None`` preserves the existing bidirectional behavior.
    """

    def __init__(
        self,
        feature_dim: int,
        nhead: int,
        dim_feedforward: int,
        clique_size: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.clique_size = clique_size

        self.seq_layer = AxisTransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
        )
        self.feat_layer = AxisTransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
        )
        self.neighbor_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.neighbor_linear1 = nn.Linear(feature_dim, dim_feedforward)
        self.neighbor_linear2 = nn.Linear(dim_feedforward, feature_dim)
        self.neighbor_dropout = nn.Dropout(dropout)
        self.neighbor_dropout1 = nn.Dropout(dropout)
        self.neighbor_dropout2 = nn.Dropout(dropout)
        self.neighbor_norm1 = nn.LayerNorm(feature_dim, eps=layer_norm_eps)
        self.neighbor_norm2 = nn.LayerNorm(feature_dim, eps=layer_norm_eps)

    # Feature index of agent_emb within the stacked feature axis (loc=0, start=1, stop=2, agent=3, cat=4).
    AGENT_FEAT_IDX = 3

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        neighbor_padding_mask: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_feats, clique_size, _ = x.shape
        if clique_size != self.clique_size:
            raise ValueError(
                f"Expected clique_size={self.clique_size}, but got {clique_size}."
            )
        if padding_mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"padding_mask must have shape {(batch_size, seq_len)}, got {tuple(padding_mask.shape)}"
            )

        # Attend over features; request weights to measure agent_emb's contribution via attention.
        # attn_weights shape: (batch*seq*clique, num_feats, num_feats); [:, :, AGENT_FEAT_IDX] is the
        # column of weights assigned to agent_emb as key across all query positions.
        x, feat_attn_weights = self.feat_layer(x, axis=2, need_weights=True)
        mean_agent_feat_attn = feat_attn_weights[:, :, self.AGENT_FEAT_IDX].mean()

        # Attend over neighbors with cross attention: the agent attends to its neighbors, but neighbors do not attend to each other.
        # neighbor_padding_mask: (B, S, C) where True = padded slot; expand to (B*S*F, C) for MHA.
        x_agent = x[:, :, :, :1, :].reshape(-1, 1, x.shape[-1])
        x_neighbors = x.reshape(-1, clique_size, x.shape[-1])
        neighbor_kpm = (
            neighbor_padding_mask[:, :, None, :]
            .expand(batch_size, seq_len, num_feats, clique_size)
            .reshape(-1, clique_size)
        )
        attn_out = self.neighbor_attn(
            query=x_agent,
            key=x_neighbors,
            value=x_neighbors,
            key_padding_mask=neighbor_kpm,
            need_weights=False,
            is_causal=False,
        )[0]
        x_agent = self.neighbor_norm1(x_agent + self.neighbor_dropout1(attn_out))
        ff = self.neighbor_linear2(
            self.neighbor_dropout(F.relu(self.neighbor_linear1(x_agent)))
        )
        x_agent = self.neighbor_norm2(x_agent + self.neighbor_dropout2(ff))
        x = torch.cat(
            [x_agent.reshape(batch_size, seq_len, num_feats, 1, x.shape[-1]), x[:, :, :, 1:, :]],
            dim=3,
        )

        # Attend over sequence, but only for the agent itself (first neighbor slot); neighbors are not part of the sequence attention.
        if padding_mask.dtype != torch.bool:
            padding_mask = padding_mask.bool()
        seq_key_padding_mask = padding_mask[:, None, None, :].expand(
            batch_size, num_feats, 1, seq_len
        )
        x_agent = x[:, :, :, :1, :]
        x_agent, _ = self.seq_layer(
            x_agent,
            axis=1,
            key_padding_mask=seq_key_padding_mask,
            attn_mask=attn_mask,
        )
        x = torch.cat([x_agent, x[:, :, :, 1:, :]], dim=3)
        return x, mean_agent_feat_attn


class HumorLayerNoNeighbor(nn.Module):
    """HumorLayer with the neighbor-attention sub-layer removed.

    Each block applies only feature attention and sequence attention
    (2 TransformerEncoderLayer-equivalents instead of 3).  Use 1.5× as many
    layers via HumorStackNoNeighbor to preserve the total depth.
    """

    def __init__(
        self,
        feature_dim: int,
        nhead: int,
        dim_feedforward: int,
        clique_size: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.clique_size = clique_size
        self.seq_layer = AxisTransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
        )
        self.feat_layer = AxisTransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
        )

    AGENT_FEAT_IDX = 3

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        neighbor_padding_mask: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_feats, clique_size, _ = x.shape

        x, feat_attn_weights = self.feat_layer(x, axis=2, need_weights=True)
        mean_agent_feat_attn = feat_attn_weights[:, :, self.AGENT_FEAT_IDX].mean()

        if padding_mask.dtype != torch.bool:
            padding_mask = padding_mask.bool()
        seq_key_padding_mask = padding_mask[:, None, None, :].expand(
            batch_size, num_feats, 1, seq_len
        )
        x_agent = x[:, :, :, :1, :]
        x_agent, _ = self.seq_layer(
            x_agent,
            axis=1,
            key_padding_mask=seq_key_padding_mask,
            attn_mask=attn_mask,
        )
        x = torch.cat([x_agent, x[:, :, :, 1:, :]], dim=3)
        return x, mean_agent_feat_attn


class HumorStack(nn.Module):
    """Stacked HumorLayer blocks with Attention3D-compatible interface."""

    def __init__(
        self,
        feature_dim: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        clique_size: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                HumorLayer(
                    feature_dim=feature_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    clique_size=clique_size,
                    dropout=dropout,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        neighbor_padding_mask: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_agent_attns = []
        for layer in self.layers:
            x, mean_agent_attn = layer(
                x, padding_mask, neighbor_padding_mask, attn_mask=attn_mask,
            )
            layer_agent_attns.append(mean_agent_attn)
        # Mean across layers; shape: scalar tensor.
        mean_agent_feat_attn = torch.stack(layer_agent_attns).mean()
        # Return only the agent's own embedding, which is the first neighbor slot.
        return x[:, :, :, 0, :], mean_agent_feat_attn


class HumorStackNoNeighbor(nn.Module):
    """HumorStack variant that uses HumorLayerNoNeighbor throughout.

    Pass num_layers = ceil(1.5 * base_num_layers) to keep total
    TransformerEncoderLayer-equivalent depth the same as a full HumorStack.
    """

    def __init__(
        self,
        feature_dim: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        clique_size: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                HumorLayerNoNeighbor(
                    feature_dim=feature_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    clique_size=clique_size,
                    dropout=dropout,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        neighbor_padding_mask: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_agent_attns = []
        for layer in self.layers:
            x, mean_agent_attn = layer(
                x, padding_mask, neighbor_padding_mask, attn_mask=attn_mask,
            )
            layer_agent_attns.append(mean_agent_attn)
        mean_agent_feat_attn = torch.stack(layer_agent_attns).mean()
        return x[:, :, :, 0, :], mean_agent_feat_attn


class TransformerStack(nn.Module):
    """Flat TransformerEncoder replacement for HumorStack.

    Default mode: take only the agent of interest (clique index 0) — co-located
    neighbors are dropped entirely, so this is a true "no co-occurrence" baseline.
    Produces a (B, S*F, D) sequence of S * num_feats = 160 tokens (with default
    S=32, F=5). Applying a standard nn.TransformerEncoder with 3 * num_humor_layers
    layers matches HumorStack depth (1 HumorLayer ≡ seq + feat + neighbor = 3
    TransformerEncoderLayers). Output shape is (B, S, F, D), identical to HumorStack.

    agent_only_concat mode (--agent_only_concat): same "agent only" semantics, but
    concatenate feature embeddings into a single vector of dimension model_dim per
    timestep, producing a (B, S, model_dim) sequence. The encoder uses d_model=model_dim.
    """

    def __init__(
        self,
        feature_dim: int,
        nhead: int,
        dim_feedforward: int,
        num_humor_layers: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        agent_only_concat: bool = False,
        model_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.agent_only_concat = agent_only_concat
        self.feature_dim = feature_dim
        n_layers = 3 * num_humor_layers
        if agent_only_concat:
            if model_dim is None:
                raise ValueError("model_dim must be provided when agent_only_concat=True")
            d_model = model_dim
        else:
            d_model = feature_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        neighbor_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_feats, clique_size, feature_dim = x.shape

        if self.agent_only_concat:
            # Take only agent of interest (clique index 0), concatenate features: (B, S, F*D).
            x_agent = x[:, :, :, 0, :]  # (B, S, F, D)
            x_flat = x_agent.reshape(batch_size, seq_len, num_feats * feature_dim)

            # padding_mask (B, S): one token per timestep.
            kpm = padding_mask.bool() if padding_mask.dtype != torch.bool else padding_mask

            out = self.encoder(x_flat, src_key_padding_mask=kpm)  # (B, S, F*D)

            # Reshape back to (B, S, F, D) — identical output shape to HumorStack.
            out = out.reshape(batch_size, seq_len, num_feats, feature_dim)
        else:
            # Take only the agent of interest (clique index 0): drop co-located
            # neighbors entirely so this is a true no-co-occurrence baseline.
            # Tokens: (B, S*F, D).
            x_agent = x[:, :, :, 0, :]  # (B, S, F, D)
            x_flat = x_agent.reshape(batch_size, seq_len * num_feats, feature_dim)

            # Expand padding_mask (B, S) -> (B, S*F): True = padded.
            kpm = (
                padding_mask[:, :, None]
                .expand(batch_size, seq_len, num_feats)
                .reshape(batch_size, seq_len * num_feats)
            )

            out = self.encoder(x_flat, src_key_padding_mask=kpm)

            # Reshape back to (B, S, F, D) — identical output shape to HumorStack.
            out = out.reshape(batch_size, seq_len, num_feats, feature_dim)

        dummy_attn = x.new_zeros(())
        return out, dummy_attn
