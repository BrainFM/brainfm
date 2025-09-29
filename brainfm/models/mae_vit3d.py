import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torchinfo import summary
from typing import NamedTuple

from brainfm.utils import to_3tuple, count_parameters
from .layer import LearnablePositionalEmbedding3D, ConditionalLayerNorm
from .encoder import ModalityAdaptedEncoder
from .decoder import MAEDecoder


class MaskingOutput(NamedTuple):
    tokens_kept: torch.Tensor      # (B, L_keep, D)
    mask_map: torch.Tensor         # (B, L_full) 0=keep, 1=masked (original order)
    restore_idx: torch.Tensor      # (B, L_full)
    keep_idx: torch.Tensor         # (B, L_keep)

class BrainFM(nn.Module):
    def __init__(self,
                 # Patch / Input
                 img_size=(128, 128, 128), # Example D, H, W
                 patch_size=(16, 16, 16),
                 patch_embed_dim=768, # Encoder/Decoder dimension (image embedding dim)
                 modality_embed_dim=768, # BioBERT output dim (text embedding dim)
                 max_patch=(20,20,20), # Max number of patches per dimension
                 # Encoder
                 encoder_depth=12,
                 encoder_nhead=12,
                 encoder_ff_dim=3072, # Typically mlp_ratio * patch_embed_dim (4 * patch_embed_dim)
                 # Decoder
                 decoder_depth=8,
                 decoder_nhead=16, # Can differ from encoder
                 decoder_ff_dim=1024,
                 # MAE
                 mask_ratio=0.75,
                 # Dropout
                 dropout=0.1):
        super().__init__()

        self.img_size   = to_3tuple(img_size)
        self.patch_size = to_3tuple(patch_size)
        self.max_patch  = to_3tuple(max_patch)

        # --- Model Parameters ---
        self.patch_embed_dim = patch_embed_dim
        self.modality_embed_dim = modality_embed_dim
        self.mask_ratio = mask_ratio
        self.num_patches_per_dim = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
            self.img_size[2] // self.patch_size[2]
        )
        self.patch_dim = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]

        # --- Input Embeddings ---
        # Projects flattened patches to patch_embed_dim
        self.patch_proj = nn.Linear(self.patch_dim, patch_embed_dim)
        
        # Project modality embedding if dims don't match patch_embed_dim
        self.modality_proj = nn.Linear(modality_embed_dim, patch_embed_dim) if modality_embed_dim != patch_embed_dim else nn.Identity()
        
        # Positional embedding (max dimensions across dataset)
        max_d, max_h, max_w = self.max_patch
        self.pos_embed = LearnablePositionalEmbedding3D(max_d, max_h, max_w, patch_embed_dim)

        # --- Learnable Tokens ---
        self.mask_token = nn.Parameter(torch.zeros(1, 1, patch_embed_dim))

        # --- Encoder ---
        self.encoder = ModalityAdaptedEncoder(
            num_layers=encoder_depth,
            d_model=patch_embed_dim,
            nhead=encoder_nhead,
            dim_feedforward=encoder_ff_dim,
            dropout=dropout,
            cond_dim=patch_embed_dim # Condition is projected modality emb
        )

        # --- Decoder ---
        self.decoder = MAEDecoder(
            num_layers=decoder_depth,
            d_model=patch_embed_dim,
            nhead=decoder_nhead,
            dim_feedforward=decoder_ff_dim,
            dropout=dropout
        )
        # Projects decoder output back to patch dimension for reconstruction
        self.decoder_pred = nn.Linear(patch_embed_dim, self.patch_dim)

        # --- Initialization ---
        self.initialize_weights()


    def initialize_weights(self):
        # Timm's trunc_normal_ initialization
        nn.init.trunc_normal_(self.pos_embed.pos_embed, std=.02)
        nn.init.trunc_normal_(self.mask_token, std=.02)
        # Initialize other layers (Linear, LayerNorm)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, ConditionalLayerNorm):
             # CLN betas already init near 0, gammas near 1 in its definition
            if hasattr(m, 'bias') and m.bias is not None:
                 nn.init.constant_(m.bias, 0)
            if hasattr(m, 'weight') and m.weight is not None:
                 nn.init.constant_(m.weight, 1.0)


    def build_input_embeddings(self, patches, modality_embeddings, position_indices):
        """
        Projects patches, adds modality conditioning and positional embeddings.

        Args:
            patches (torch.Tensor): [B, L, PatchDim]
            modality_embeddings (torch.Tensor): [B, L, ModEmbDim]
            position_indices (torch.Tensor): [B, L, 3], use -1 for padded entries

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                x (B, L, D): combined embeddings
                mod_cond (B, L, D): projected modality embeddings (for CLN/conditioning)
                pos_embed (B, L, D): positional embeddings used (zero on pads)
        """
        x = self.patch_proj(patches)                             # (B, L, D)
        mod_cond = self.modality_proj(modality_embeddings)       # (B, L, D)
        x = x + mod_cond
        pos_embed = self.pos_embed(position_indices)                 # (B, L, D), zero on pads
        x = x + pos_embed
        return x, mod_cond, pos_embed


    def apply_random_masking(self, x, pad_mask):
        """Random masking respecting padding; keeps the same L_keep across batch.

        Args:
            x (torch.Tensor): (B, L, D)
            pad_mask (torch.Tensor): (B, L) with True at padded positions

        Returns:
            MaskingOutput: tokens_kept, mask_map (0=keep,1=masked, original order), restore_idx, keep_idx
        """
        B, L, D = x.shape
        is_valid = ~pad_mask                       # (B, L)
        n_valid = is_valid.sum(dim=1)              # (B,)

        k_per_sample = (n_valid.float() * (1.0 - self.mask_ratio)).round().clamp(min=1).to(torch.long)
        L_keep = int(k_per_sample.min().item())

        noise = torch.rand(B, L, device=x.device)
        noise.masked_fill_(pad_mask, float('inf'))  # pads sorted last

        shuffle_idx = torch.argsort(noise, dim=1)   # ascending: kept first
        restore_idx = torch.argsort(shuffle_idx, dim=1)
        keep_idx = shuffle_idx[:, :L_keep]

        tokens_kept = torch.gather(x, dim=1, index=repeat(keep_idx, 'b l -> b l d', d=D))

        mask_map = torch.ones((B, L), device=x.device)
        mask_map[:, :L_keep] = 0
        mask_map = torch.gather(mask_map, dim=1, index=restore_idx)  # back to original order

        # Debug invariant: kept tokens must be non-pad
        if torch.is_grad_enabled():
            assert (~pad_mask).gather(1, keep_idx).all(), "keep_idx must select only non-pad tokens."

        return MaskingOutput(tokens_kept, mask_map, restore_idx, keep_idx)


    def build_decoder_input(self, enc_out, pos_full, restore_idx, mask_map, mod_cond_full):
        """Construct MAE decoder input in ORIGINAL order.

        Args:
            enc_out (torch.Tensor): (B, L_keep, D)
            pos_full (torch.Tensor): (B, L_full, D)
            restore_idx (torch.Tensor): (B, L_full)
            mask_map (torch.Tensor): (B, L_full), 0=keep, 1=masked
            mod_cond_full (torch.Tensor): (B, L_full, D)

        Returns:
            torch.Tensor: decoder input (B, L_full, D)
        """
        B, L_keep, D = enc_out.shape
        _, L_full, _ = pos_full.shape
        L_mask = L_full - L_keep

        mask_tokens_base = self.mask_token.repeat(B, L_mask, 1)  # (B, L_mask, D)
        mask_bool = (mask_map == 1)                               # (B, L_full)
        # Invariant: exactly L_mask masked tokens per sample
        assert (mask_bool.sum(dim=1) == L_mask).all(), "mask_map must mark exactly L_mask tokens per sample."
        # Column indices of masked tokens in original order
        masked_idx = torch.nonzero(mask_bool, as_tuple=False)[:, 1].view(B, L_mask)  # (B, L_mask)
        modality_cond_masked_original_order = torch.gather(
            mod_cond_full, dim=1,
            index=repeat(masked_idx, 'b l -> b l d', d=D)
        )  # (B, L_mask, D)
        conditioned_mask_tokens = mask_tokens_base + modality_cond_masked_original_order  # (B, L_mask, D)

        decoder_input_shuffled = torch.cat([enc_out, conditioned_mask_tokens], dim=1)  # (B, L_full, D)
        decoder_input_original_order = torch.gather(
            decoder_input_shuffled, dim=1, index=repeat(restore_idx, 'b l -> b l d', d=D)
        )
        return decoder_input_original_order + pos_full
    
    def masked_recon_loss(
        self,
        patches: torch.Tensor,        # (B, L_full, P)
        recon: torch.Tensor,          # (B, L_full, P)
        pad_mask: torch.Tensor,       # (B, L_full) True=PAD
        mask_map: torch.Tensor,       # (B, L_full) 0=keep, 1=masked (original order)
        clamp: tuple | None = None    # e.g., (0.0, 1.0) for min-max data; None for z-score
    ) -> torch.Tensor:
        """Compute MAE reconstruction loss over masked & non-padded tokens only.
        - Leaves TARGETS untouched (no bias).
        - Sanitizes PREDICTIONS to avoid NaN/Inf backprop.
        - Ignores non-finite elements with an elementwise mask.
        """
        is_valid = ~pad_mask                          # (B, L_full)
        loss_mask = (mask_map == 1) & is_valid        # (B, L_full)
        if not loss_mask.any():
            return torch.tensor(0.0, device=patches.device, requires_grad=True)

        target = patches[loss_mask]                   # (N_masked, P)
        pred   = recon[loss_mask]                     # (N_masked, P)

        # Elementwise finite mask (do NOT modify targets)
        elem_mask = torch.isfinite(target) & torch.isfinite(pred)
        if not elem_mask.any():
            return torch.tensor(0.0, device=patches.device, requires_grad=True)

        # Sanitize predictions
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0)
        if clamp is not None:
            pred = torch.clamp(pred, clamp[0], clamp[1])

        return F.mse_loss(pred[elem_mask], target[elem_mask], reduction='mean')


    def forward_pretraining_step(self, patches, modality_embeddings, position_indices, pad_mask):
        """Forward pass for MAE pre-training."""
        # 1. Build input embeddings (also get positional embeddings to reuse)
        x, mod_cond, pos_full = self.build_input_embeddings(patches, modality_embeddings, position_indices)

        # 2. Apply masking
        masking = self.apply_random_masking(x, pad_mask)
        tokens_kept, mask_map, restore_idx, keep_idx = masking

        # Gather modality conditions for kept tokens (encoder input)
        D = x.size(-1)
        mod_cond_kept = torch.gather(mod_cond, dim=1, index=repeat(keep_idx, 'b l -> b l d', d=D))
        enc_pad_mask = torch.gather(pad_mask, dim=1, index=keep_idx)

        # 3. Encode kept tokens
        enc_out = self.encoder(tokens_kept, src_key_padding_mask=enc_pad_mask, cond=mod_cond_kept)

        # 4. Build decoder input (original order)
        dec_in = self.build_decoder_input(enc_out, pos_full, restore_idx, mask_map, mod_cond)
        dec_pad_mask = pad_mask

        # 5. Decode
        dec_out = self.decoder(
            tgt=dec_in,
            memory=enc_out,
            tgt_key_padding_mask=dec_pad_mask,
            memory_key_padding_mask=enc_pad_mask
        )

        # 6. Predict and compute loss on masked, non-padded tokens
        recon = self.decoder_pred(dec_out)  # (B, L_full, P)
        loss = self.masked_recon_loss(
            patches=patches,
            recon=recon,
            pad_mask=pad_mask,
            mask_map=mask_map,
            clamp=(0.0, 1.0))  # set None if z-score
        return loss


    def forward_finetune_step(self, patches, modality_embeddings, position_indices, pad_mask, downstream_head):
        """Forward pass for fine-tuning."""
        x, mod_cond, _ = self.build_input_embeddings(patches, modality_embeddings, position_indices)
        enc_out = self.encoder(x, src_key_padding_mask=pad_mask, cond=mod_cond)
        return downstream_head(enc_out, attention_mask=pad_mask)


    def forward(self, patches, modality_embeddings, position_indices, pad_mask, downstream_head=None):
        # Pre-training
        if downstream_head is None:
            return self.forward_pretraining_step(
                patches=patches,
                modality_embeddings=modality_embeddings,
                position_indices=position_indices,
                pad_mask=pad_mask,
            )
        # Fine-tuning
        else:
            return self.forward_finetune_step(
                patches=patches,
                modality_embeddings=modality_embeddings,
                position_indices=position_indices,
                pad_mask=pad_mask,
                downstream_head=downstream_head,
            )
        
    def patchify_from_volumes(self, x):
        """
        x: (B, M, D, H, W)
        Returns:
            patches: (B, M*n_patches, P) where P = p_d*p_h*p_w
            n_pd, n_ph, n_pw: ints for grid size
        """
        p_d, p_h, p_w = self.patch_size
        B, M, D, H, W = x.shape
        assert D % p_d == 0 and H % p_h == 0 and W % p_w == 0, "Input must be divisible by patch_size"
        n_pd, n_ph, n_pw = D // p_d, H // p_h, W // p_w
        patches = rearrange(x, 'b m (d pd) (h ph) (w pw) -> b (m d h w) (pd ph pw)', pd=p_d, ph=p_h, pw=p_w)
        return patches, n_pd, n_ph, n_pw

    def expand_modality_and_positions(self, B, M, n_pd, n_ph, n_pw, modality_embs):
        """
        modality_embs: (B, M, EmbDim) or (M, EmbDim) broadcastable to (B, M, EmbDim)
        Returns:
            mod_token_embs: (B, M*n_patches, EmbDim)
            pos_indices:    (B, M*n_patches, 3) with (d,h,w)
        """
        L = n_pd * n_ph * n_pw
        if modality_embs.ndim == 2:
            modality_embs = modality_embs.unsqueeze(0).expand(B, -1, -1)  # (B, M, E)
        # repeat each modality embedding L times along the token dimension
        mod_token_embs = repeat(modality_embs, 'b m e -> b (m l) e', l=L)
        d = torch.arange(n_pd, device=modality_embs.device)
        h = torch.arange(n_ph, device=modality_embs.device)
        w = torch.arange(n_pw, device=modality_embs.device)
        Dg, Hg, Wg = torch.meshgrid(d, h, w, indexing='ij')  # (n_pd, n_ph, n_pw)
        pos = torch.stack([Dg, Hg, Wg], dim=-1).reshape(1, L, 3)  # (1, L, 3)
        pos = repeat(pos, '1 l c -> b (m l) c', b=B, m=M)         # (B, M*L, 3)
        return mod_token_embs, pos

    def forward_from_volumes(self, images, modality_embs, modality_mask=None, downstream_head=None):
        """
        images: (B, M, D, H, W) CPU/GPU float
        modality_embs: (B, M, Em) or (M, Em)
        modality_mask: (B, M) bool or None. True=PAD. Used to mask modalities.
        """
        patches, n_pd, n_ph, n_pw = self.patchify_from_volumes(images)
        B = patches.size(0)
        M = images.size(1)
        mod_tok, pos_idx = self.expand_modality_and_positions(B, M, n_pd, n_ph, n_pw, modality_embs.to(patches.device))
        L = n_pd * n_ph * n_pw
        if modality_mask is None:
            pad_mask = torch.zeros(B, M * L, dtype=torch.bool, device=patches.device)
        else:
            pad_mask = repeat(modality_mask.to(patches.device), 'b m -> b (m l)', l=L)
        if downstream_head is None:
            return self.forward(patches, mod_tok, pos_idx, pad_mask)
        else:
            return self.forward(patches, mod_tok, pos_idx, pad_mask, downstream_head=downstream_head)


def load_model_base():
    return BrainFM(
        img_size=(128, 128, 128),
        patch_size=(16, 16, 16),
        patch_embed_dim=768,
        modality_embed_dim=768,
        max_patch=(20, 20, 20),
        encoder_depth=12,
        encoder_nhead=12,
        encoder_ff_dim=3072, #(768 * 4)
        decoder_depth=8,
        decoder_nhead=12,
        decoder_ff_dim=1024,
        mask_ratio=0.75
    )

def build_model(config, device: torch.device, logger=None):
    model = load_model_base()
    model.to(device)

    # Load resume checkpoint if specified
    ckpt_path = config.paths.resume_checkpoint_path
    if ckpt_path and os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if logger:
            logger.info(f"Loaded model weights from checkpoint: {ckpt_path}")
    else:
        if logger:
            logger.info(f"Checkpoint not found. Training from scratch.")

    # Print number of parameters
    total_params = count_parameters(model)
    if logger:
        logger.info(f"Total trainable parameters: {total_params:,}")

    return model


if __name__ == "__main__":
    model = load_model_base()
    model.train()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Positional embedding shape: {model.pos_embed.pos_embed.shape}")
    print(f"Patch dimension (flattened): {model.patch_dim}")
    print(f"Number of patches per dimension: {model.num_patches_per_dim}")

    B, L = 2, 64  # batch size and number of patches
    patch_dim = model.patch_dim
    patches = torch.randn(B, L, patch_dim)
    modality_embeddings = torch.randn(B, L, model.modality_embed_dim)
    position_indices = torch.randint(low=0, high=10, size=(B, L, 3))
    pad_mask = torch.zeros(B, L, dtype=torch.bool)  # no padding

    summary(model)

    with torch.no_grad():
        loss = model(
            patches=patches,
            modality_embeddings=modality_embeddings,
            position_indices=position_indices,
            pad_mask=pad_mask,
        )
    print(f"Dummy forward loss: {loss.item():.6f}")

    # Additional dummy run for forward_from_volumes
    images = torch.randn(2, 3, 128, 128, 128)
    modality_embs = torch.randn(3, model.modality_embed_dim)
    with torch.no_grad():
        loss2 = model.forward_from_volumes(images, modality_embs)
    print(f"Dummy forward_from_volumes loss: {loss2.item():.6f}")


    # Test forward_from_volumes with modality_mask (simulate padding third modality for first sample)
    modality_mask = torch.zeros(2, 3, dtype=torch.bool)
    modality_mask[0, 2] = True  # First sample, third modality is PAD
    with torch.no_grad():
        loss3 = model.forward_from_volumes(images, modality_embs, modality_mask=modality_mask)
    print(f"Dummy forward_from_volumes (with modality_mask) loss: {loss3.item():.6f}")

    # ---- Unit test: zero-grad for padded modalities ----
    model.zero_grad()
    images = torch.randn(2, 3, 128, 128, 128, requires_grad=True)
    modality_embs = torch.randn(3, model.modality_embed_dim)
    modality_mask = torch.zeros(2, 3, dtype=torch.bool)
    modality_mask[0, 2] = True  # pad 3rd modality for sample 0
    loss = model.forward_from_volumes(images, modality_embs, modality_mask=modality_mask)
    loss.backward()
    grad_pad = images.grad[0, 2]        # (D,H,W) for padded modality
    grad_keep = images.grad[0, 0]       # compare with a kept modality
    print("Grad (padded modality) L2:", float(grad_pad.pow(2).sum().sqrt()))
    print("Grad (kept modality)   L2:", float(grad_keep.pow(2).sum().sqrt()))
    assert torch.allclose(grad_pad, torch.zeros_like(grad_pad)), "Padded modality should have zero gradients."
    print("Unit test passed: padded modality receives zero gradient.")
