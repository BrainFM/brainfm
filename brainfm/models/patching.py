import torch
import torch.nn as nn
from einops import rearrange


# --- Patching ---
def patchify_3d(imgs, patch_size):
    """
    imgs: (B, M, D, H, W)
    patch_size: tuple (p_d, p_h, p_w)
    output: (B, M * n_patches, p_d * p_h * p_w)
            where n_patches = (D // p_d) * (H // p_h) * (W // p_w)
    """
    B, M, D, H, W = imgs.shape
    p_d, p_h, p_w = patch_size
    # NOTE: imgs expects channels as modalities (M). Shapes must be exactly divisible by patch_size.
    assert imgs.shape[2] % p_d == 0 and imgs.shape[3] % p_h == 0 and imgs.shape[4] % p_w == 0

    n_pd = D // p_d
    n_ph = H // p_h
    n_pw = W // p_w

    # Rearrange to extract patches:
    # From shape: (B, M, D, H, W)
    # To shape: (B, M, n_patches, patch_volume) where patch_volume = p_d*p_h*p_w.
    x = rearrange(imgs, 'b m (d pd) (h ph) (w pw) -> b m (d h w) (pd ph pw)',
                  pd=p_d, ph=p_h, pw=p_w)
    # Merge modality and patch dimensions: (B, M, n_patches, patch_volume) -> (B, M*n_patches, patch_volume)
    x = rearrange(x, 'b m np p_vol -> b (m np) p_vol')
    
    return x, (M, n_pd, n_ph, n_pw, p_d, p_h, p_w)


def unpatchify_3d(x, patch_info):
    """
    x: (B, M * n_patches, p_d*p_h*p_w)
    patch_info: tuple (M, n_pd, n_ph, n_pw, p_d, p_h, p_w)
    output: (B, M, D, H, W)
             where D = n_patches_d * p_d, H = n_patches_h * p_h, W = n_patches_w * p_w
    """
    # NOTE: This is the exact inverse of patchify_3d for divisible shapes; it assumes the same (M, n_pd, n_ph, n_pw, p_d, p_h, p_w).
    M, n_pd, n_ph, n_pw, p_d, p_h, p_w = patch_info
    # First, reshape to separate modalities from patches:
    x = rearrange(x, 'b (m np) p_vol -> b m np p_vol', m=M)
    # Now, we know np = n_pd * n_ph * n_pw; rearrange to recover spatial dimensions.
    x = rearrange(x, 'b m (d h w) (pd ph pw) -> b m (d pd) (h ph) (w pw)', 
                  d=n_pd, h=n_ph, w=n_pw, pd=p_d, ph=p_h, pw=p_w)
    return x
