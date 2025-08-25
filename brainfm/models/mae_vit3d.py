import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from torchinfo import summary

from brainfm.utils import to_3tuple, count_parameters
from .layer import LearnablePositionalEmbedding3D, ConditionalLayerNorm
from .encoder import ModalityAdaptedEncoder
from .decoder import MAEDecoder

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


    def _prepare_input_embeddings(self, patches, modality_embeddings, position_indices):
        """
        Projects patch embeddings, adds modality embeddings, and adds
        positional embeddings (handles padding correctly via self.pos_embed).

        Args:
            patches (torch.Tensor): Input patches [B, SeqLen, PatchDim]
            modality_embeddings (torch.Tensor): Modality embeddings [B, SeqLen, ModEmbDim]
            position_indices (torch.Tensor): 3D indices for each patch [B, SeqLen, 3].
                                             Padding should use values like -1.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Combined input embeddings [B, SeqLen, EmbedDim]
                - Projected modality embeddings [B, SeqLen, EmbedDim] (returned separately if needed later)
        """
        # 1. Project patches to the embedding dimension
        # x shape: (B, SeqLen, EmbedDim)
        x = self.patch_proj(patches)

        # 2. Project modality embeddings (if needed) and add them
        # mod_emb_proj shape: (B, SeqLen, EmbedDim)
        mod_emb_proj = self.modality_proj(modality_embeddings)
        x = x + mod_emb_proj

        # 3. Get positional embeddings
        # self.pos_embed handles padding internally, returning zeros for padded indices.
        # pos_embed shape: (B, SeqLen, EmbedDim)
        pos_embed = self.pos_embed(position_indices)

        # 4. Add positional embeddings directly
        x = x + pos_embed

        # Return the combined embeddings and optionally the projected modality embeddings
        # if they are needed later (e.g., for Conditional Layer Norm)
        return x, mod_emb_proj


    def _mask_tokens(self, x, attention_mask):
        """ Perform random masking (padding-aware); keep the same number of tokens across the batch. """
        B, L, D = x.shape

        # valid tokens (False = pad, True = valid)
        is_valid = ~attention_mask  # shape (B, L)

        # per-sample valid counts
        n_valid = is_valid.sum(dim=1)  # (B,)

        # compute per-sample desired keep, then choose a batch-wide keep that is safe for all samples
        k_per_sample = (n_valid.float() * (1.0 - self.mask_ratio)).round().clamp(min=1).to(torch.long)
        len_keep = int(k_per_sample.min().item())

        # random noise; push pads to +inf so they are ranked last
        noise = torch.rand(B, L, device=x.device)
        noise.masked_fill_(attention_mask, float('inf'))

        # sort noise ascending -> kept tokens come first
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # take the first len_keep indices per sample
        ids_keep = ids_shuffle[:, :len_keep]

        # gather kept tokens
        x_masked = torch.gather(x, dim=1, index=repeat(ids_keep, 'b l -> b l d', d=D))

        # build mask in shuffled order: 0 for kept, 1 for removed
        mask = torch.ones((B, L), device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to original order
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep


    def _prepare_decoder_input(self, encoder_output, # (B, L_unmasked, D)
                               pos_embed_full,       # (B, L_full, D) - Positional embeds for ALL original positions
                               ids_restore,          # (B, L_full) - Indices to unshuffle
                               mask_logic,           # (B, L_full) -> 0=keep, 1=mask (original order)
                               modality_cond_full):  # (B, L_full, D) <- FULL projected modality conditions
        """ Construct the input sequence for the MAE decoder IN ORIGINAL ORDER """
        B, L_unmasked, D = encoder_output.shape
        _, L_full, _ = pos_embed_full.shape
        num_masked = L_full - L_unmasked

        # --- 1. Prepare conditioning for the MASKED tokens ---
        # Create learnable mask tokens base
        mask_tokens_base = self.mask_token.repeat(B, num_masked, 1) # (B, L_masked, D)

        # Select the ORIGINAL modality conditions ONLY for the masked positions
        # We need to select based on mask_logic == 1 and reshape correctly
        # Ensure mask_logic is boolean for indexing
        mask_bool = mask_logic == 1 # (B, L_full), True where masked
        modality_cond_masked_original_order = modality_cond_full[mask_bool].view(B, num_masked, D) # (B, L_masked, D)

        # Combine base mask token with its original modality
        conditioned_mask_tokens = mask_tokens_base + modality_cond_masked_original_order # (B, L_masked, D)


        # --- 2. Combine Encoder Output (unmasked) and Conditioned Mask Tokens (masked) IN SHUFFLED ORDER ---
        # This follows the original MAE paper's approach: concatenate the kept tokens
        # (encoder output) and the replacement tokens (conditioned mask tokens)
        decoder_input_shuffled = torch.cat([encoder_output, conditioned_mask_tokens], dim=1) # (B, L_full, D)


        # --- 3. Unshuffle the combined sequence back to the original patch order ---
        decoder_input_original_order = torch.gather(
            decoder_input_shuffled,
            dim=1,
            index=repeat(ids_restore, 'b l -> b l d', d=D)
        ) # (B, L_full, D)


        # --- 4. Add the FULL positional embeddings ---
        # Positional embeddings correspond to the original order, so add after unshuffling
        decoder_input_final = decoder_input_original_order + pos_embed_full # (B, L_full, D)

        return decoder_input_final


    def forward_pretrain(self, patches, modality_embeddings, position_indices,
                         attention_mask):
        """ Forward pass for MAE pre-training """

        # 1. Get input embeddings (patch + modality + position)
        # x: (B, L, D) - combined patch+modality+pos
        # modality_cond: (B, L, D) - projected modality embedding only
        x, modality_cond = self._prepare_input_embeddings(patches, modality_embeddings, position_indices)
        B, L, D = x.shape

        # Store full positional embedding before masking
        pos_embed_full = self.pos_embed(position_indices)

        # 2. Perform masking
        # x_unmasked: (B, L_unmasked, D) -> tokens fed to encoder
        # mask_logic: (B, L) -> 0 for keep, 1 for mask (original order)
        # ids_restore: (B, L) -> indices to unshuffle
        # ids_keep: (B, L_unmasked) -> indices of kept tokens in original order
        x_unmasked, mask_logic, ids_restore, ids_keep = self._mask_tokens(x, attention_mask)

        # Also gather corresponding modality conditions for unmasked tokens for the ENCODER input
        modality_cond_unmasked = torch.gather(
            modality_cond,
            dim=1,
            index=repeat(ids_keep, 'b l -> b l d', d=D)
        )

        # Create key padding mask for encoder (True where padding in the UNMASKED sequence)
        # Gather the original attention mask based on kept indices
        encoder_padding_mask = torch.gather(attention_mask, dim=1, index=ids_keep) # Shape: (B, L_unmasked)


        # 3. Encode unmasked tokens
        encoder_output = self.encoder(x_unmasked,
                                      src_key_padding_mask=encoder_padding_mask,
                                      cond=modality_cond_unmasked) # (B, L_unmasked, D)


        # 4. Prepare decoder input
        decoder_input = self._prepare_decoder_input(
            encoder_output=encoder_output,        # Output for unmasked tokens
            pos_embed_full=pos_embed_full,        # Positional embeds for ALL original positions
            ids_restore=ids_restore,              # Indices to unshuffle
            mask_logic=mask_logic,                # Mask indicating which original positions were masked (0=keep, 1=mask)
            modality_cond_full=modality_cond      # << NEW: Pass the FULL projected modality conditions
        )

        # Create key padding mask for decoder (True where padding in the FULL sequence)
        decoder_padding_mask = attention_mask # Use original full sequence padding mask


        # 5. Decode
        # The decoder now operates on the full sequence reconstructed in the original order
        decoder_output = self.decoder(
            tgt=decoder_input,                    # Full sequence with replacements
            memory=encoder_output,                # Encoder output (serves as context/memory)
            tgt_key_padding_mask=decoder_padding_mask, # Mask padding in the target (full sequence)
            memory_key_padding_mask=encoder_padding_mask # Mask padding in the memory (unmasked sequence)
        )


        # 6. Predict masked patches
        reconstruction = self.decoder_pred(decoder_output) # (B, L_full, PatchDim)

        # --- Calculate Loss ---
        target_patches = patches # Original patches (B, L, PatchDim)

        # Calculate loss only for masked, non-padded patches
        is_valid = ~attention_mask  # True where real tokens
        mask_for_loss = (mask_logic == 1) & is_valid

        valid_target_patches = target_patches[mask_for_loss]
        valid_reconstruction = reconstruction[mask_for_loss]

        if valid_target_patches.numel() > 0:
            loss = F.mse_loss(valid_reconstruction, valid_target_patches, reduction='mean')
        else:
            loss = torch.tensor(0.0, device=patches.device, requires_grad=True)


        # Optional: Return reconstruction for visualization
        # We need to unpatchify reconstruction[mask_logic == 1] which is complex
        # Easier to return full reconstruction and mask
        return loss #, reconstruction, mask_logic # Return only loss usually


    def forward_finetune(self, patches, modality_embeddings, position_indices,
                         attention_mask, downstream_head):
        """ Forward pass for fine-tuning """
         # 1. Get input embeddings (patch + modality + position)
        x, modality_cond = self._prepare_input_embeddings(patches, modality_embeddings, position_indices)

        # 2. Encode *all* tokens (no masking)
        # Use the provided attention_mask for padding
        encoder_output = self.encoder(x,
                                      src_key_padding_mask=attention_mask,
                                      cond=modality_cond) # (B, SeqLen, D)

        # 3. Apply downstream head
        # Example: Use average pooling or a [CLS] token if added
        # Here, just pass the full sequence to the head (head needs to handle it)
        output = downstream_head(encoder_output, attention_mask=attention_mask)

        return output


    def forward(self, patches, modality_embeddings, position_indices,
                attention_mask, downstream_head=None):

        # Pre-training
        if downstream_head is None:
            return self.forward_pretrain(
                patches=patches,
                modality_embeddings=modality_embeddings,
                position_indices=position_indices,
                attention_mask=attention_mask,
            )
        
        # Fine-tuning
        else:
            return self.forward_finetune(
                patches=patches,
                modality_embeddings=modality_embeddings,
                position_indices=position_indices,
                attention_mask=attention_mask,
                downstream_head=downstream_head
            )
        
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
    ckpt_path = config.train.resume_checkpoint_path
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

    # Print model summary
    summary(model) 
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

    from torchinfo import summary
    summary(model) 