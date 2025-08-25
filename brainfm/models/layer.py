import torch
import torch.nn as nn
from einops import rearrange

# --- Positional Embedding ---
class LearnablePositionalEmbedding3D(nn.Module):
    def __init__(self, max_patches_d, max_patches_h, max_patches_w, embed_dim):
        """
        Learnable 3D Positional Embedding grid.

        Args:
            max_patches_d (int): Max depth dimension of the patch grid.
            max_patches_h (int): Max height dimension of the patch grid.
            max_patches_w (int): Max width dimension of the patch grid.
            embed_dim (int): The embedding dimension.
        """
        super().__init__()
        # Learnable parameter grid for positional embeddings
        # Shape: (1, Depth, Height, Width, EmbedDim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_patches_d, max_patches_h, max_patches_w, embed_dim)
        )
        # Using nn.Parameter registers it for learning during training
        nn.init.trunc_normal_(self.pos_embed, std=.02) # Initialize embeddings

    def forward(self, position_indices):
        """
        Looks up positional embeddings based on 3D indices.
        Handles padding/out-of-range indices by returning zero vectors.
        
        Args:
            position_indices (torch.Tensor): shape (B, SeqLen, 3) with (d,h,w) per patch.
                Padded or invalid entries should contain negative numbers or indices outside the
                range [0, D/H/W). These will be zeroed out in the returned embeddings.
        Returns:
            torch.Tensor: Positional embeddings of shape (B, SeqLen, embed_dim),
                          with zero vectors for invalid (padded/out-of-bounds) positions.
        """
        # --- Safety checks ---
        assert position_indices.dtype in (torch.int32, torch.int64), (
            "position_indices must be integer dtype (int32 or int64).")

        B, seq_len, _ = position_indices.shape
        _, D, H, W, embed_dim = self.pos_embed.shape

        # Extract (d, h, w) indices
        d_indices = position_indices[..., 0]
        h_indices = position_indices[..., 1]
        w_indices = position_indices[..., 2]

        # --- Identify valid (non-padded & in-bounds) positions ---
        # Valid if indices are within [0, size) for each dimension
        valid_pos_mask = (
            (d_indices >= 0) & (d_indices < D) &
            (h_indices >= 0) & (h_indices < H) &
            (w_indices >= 0) & (w_indices < W)
        )  # shape: (B, SeqLen)

        # --- Compute flat indices for gathering ---
        # Note: we'll clamp indices for safe gather, then zero out invalids using valid_pos_mask.
        flat_pos_embed = self.pos_embed.view(1, D * H * W, embed_dim).expand(B, -1, -1)
        flat_indices = d_indices * (H * W) + h_indices * W + w_indices  # (B, SeqLen)
        flat_indices_clamped = torch.clamp(flat_indices, min=0, max=D * H * W - 1)
        indices_for_gather = flat_indices_clamped.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Gather embeddings; invalids will be masked to zero next
        gathered_embed = torch.gather(flat_pos_embed, 1, indices_for_gather)  # (B, SeqLen, embed_dim)

        # Zero-out embeddings for invalid positions
        output_embed = torch.where(valid_pos_mask.unsqueeze(-1), gathered_embed, torch.zeros_like(gathered_embed))
        return output_embed


# --- Conditional Layer Normalization ---
class ConditionalLayerNorm(nn.Module):
    def __init__(self, normalized_shape, cond_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape # Should be embed_dim

        # Linear maps from condition -> affine params (initialized to zeros)
        # so that at initialization CLN behaves like standard LayerNorm (gamma=1, beta=0)
        self.gamma_mlp = nn.Linear(cond_dim, normalized_shape)
        self.beta_mlp  = nn.Linear(cond_dim, normalized_shape)

        nn.init.zeros_(self.gamma_mlp.weight)
        nn.init.zeros_(self.gamma_mlp.bias)
        nn.init.zeros_(self.beta_mlp.weight)
        nn.init.zeros_(self.beta_mlp.bias)


    def forward(self, x, cond):
        """
        x:    (B, SeqLen, EmbedDim)
        cond: (B, SeqLen, CondDim) or (B, 1, CondDim) — the latter will be broadcast across SeqLen.
        """
        B, S, E = x.shape
        assert cond is not None, "ConditionalLayerNorm requires a conditioning tensor."
        if cond.dim() == 2:
            # Accept (B, CondDim) and broadcast across sequence
            cond = cond.unsqueeze(1).expand(-1, S, -1)
        elif cond.size(1) == 1 and S != 1:
            # Broadcast (B,1,C) to (B,S,C)
            cond = cond.expand(-1, S, -1)
        else:
            # Ensure shapes align
            assert cond.size(0) == B and cond.size(1) == S, (
                f"cond shape {cond.shape} must match batch/seq of x {x.shape}")

        # Standard LayerNorm stats per token
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Affine params from condition; start near identity (gamma≈1, beta≈0)
        gamma = 1.0 + self.gamma_mlp(cond)
        beta  = self.beta_mlp(cond)
        return x_normalized * gamma + beta
    