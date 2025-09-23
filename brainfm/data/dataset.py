import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from brainfm.utils import to_3tuple
from brainfm.models import patchify_3d

class ModalityEncoderCache:
    """
    Small wrapper to cache text -> embedding for a Hugging Face encoder.
    Expects `encoder` to be a callable that accepts a string and returns a 1D or (1, D) tensor.
    Embeddings are stored on `cache_device` (default 'cpu').
    """
    def __init__(self, encoder, cache_device: str = "cpu"):
        self.encoder = encoder
        self.cache_device = cache_device
        self._cache = {}

    def get(self, text: str) -> torch.Tensor:
        if text in self._cache:
            return self._cache[text]
        with torch.no_grad():
            emb = self.encoder(text)
            if not isinstance(emb, torch.Tensor):
                raise TypeError(f"Encoder must return a torch.Tensor, got {type(emb)} for text '{text}'")
            if emb.ndim == 2:
                if emb.shape[0] != 1:
                    raise ValueError(f"Expected encoder to return shape (D,) or (1, D), got {tuple(emb.shape)}")
                emb = emb.squeeze(0)
            elif emb.ndim != 1:
                raise ValueError(f"Expected encoder to return 1D tensor, got shape {tuple(emb.shape)}")
            emb = emb.detach().to(self.cache_device)
        self._cache[text] = emb
        return emb

def load_nifti(path):
    """Load a NIfTI file (.nii.gz) and return a NumPy array (D, H, W or vendor default)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return nib.load(path).get_fdata()


def load_npy_dhw(path):
    """Load a .npy volume saved as (H, W, D) and return as (D, H, W)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(path).transpose(2, 0, 1)  # (H, W, D) → (D, H, W)
 

def load_modalities(modality_paths):
    """Load multiple modality volumes from file paths; keys are modality names.
    Returns a dict of {modality: torch.FloatTensor} with shape (D, H, W) on CPU.
    """
    volumes = {}
    for modality, path in modality_paths.items():
        if path.endswith('.nii.gz'):
            vol = load_nifti(path)
        elif path.endswith('.npy'):
            vol = load_npy_dhw(path)
        else:
            raise ValueError(f"Unsupported file format for {modality}: {path}")
        
        # Ensure 3D shape
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D ndarray for {modality}, got shape {vol.shape}")
        
        #  Convert to tensor float32
        if isinstance(vol, np.ndarray):
            vol = torch.from_numpy(vol).float()
        elif isinstance(vol, torch.Tensor):
            vol = vol.float()
        else:
            raise TypeError(f"Unsupported volume type for {modality}: {type(vol)}")
        
        volumes[modality] = vol
    return volumes

def validate_sample_dict(sample_dict):
    """
    Validate the sample_dict for MultiModMRIDataset.
    Ensures:
      - sample_dict is a dictionary with non-empty values.
      - Keys are strings (sample IDs).
      - Values are dictionaries mapping modality names (strings) to valid file paths (str).
      - Each file path must exist on disk.
    Raises ValueError if any check fails, with a clear message.
    Returns True if all checks pass.
    """
    if not isinstance(sample_dict, dict):
        raise ValueError("sample_dict must be a dictionary.")
    if not sample_dict:
        raise ValueError("sample_dict is empty.")
    for sample_id, modality_dict in sample_dict.items():
        if not isinstance(sample_id, str):
            raise ValueError(f"Sample ID must be a string, got {type(sample_id)}.")
        if not modality_dict:
            raise ValueError(f"No modalities found for sample_id='{sample_id}'.")
        if not isinstance(modality_dict, dict):
            raise ValueError(f"Value for sample_id '{sample_id}' must be a dict, got {type(modality_dict)}.")
        for mod_name, file_path in modality_dict.items():
            if not isinstance(mod_name, str):
                raise ValueError(f"Modality name must be a string, got {type(mod_name)} for sample_id '{sample_id}'.")
            if not os.path.isfile(file_path):
                raise ValueError(f"File path does not exist for modality '{mod_name}' in sample_id '{sample_id}': {file_path}")
            if not isinstance(file_path, str):
                raise ValueError(f"File path for modality '{mod_name}' in sample_id '{sample_id}' must be a string, got {type(file_path)}.")
    return True

class MultiModMRIDataset(Dataset):
    """
    Dataset for multi-modality 3D MRI that patchifies per modality and attaches:
      - cached modality text embeddings (computed once via a Hugging Face encoder),
      - per-patch (d,h,w) indices,
      - patch grid size for positional embeddings.

    Notes:
      * Modality embeddings are cached on CPU by default and repeated per patch.
      * Move tensors to GPU later (e.g., in your training step or collate_fn).
    """
    def __init__(self, sample_dict, modality_encoder, patch_size, precompute_modalities: bool = True, cache_device: str = "cpu"):
        
        validate_sample_dict(sample_dict)

        self.sample_dict = sample_dict
        self.sample_ids  = list(self.sample_dict.keys())
        self.patch_size  = to_3tuple(patch_size)

        # Wrap encoder with a cache (embeddings stored on CPU by default)
        self.modality_cache = ModalityEncoderCache(
            modality_encoder,
            cache_device=cache_device
        )

        # Optionally precompute all unique modality names found in the dataset
        if precompute_modalities:
            unique_modalities = set()
            for _sid, mods in self.sample_dict.items():
                unique_modalities.update(list(mods.keys()))
            for m in sorted(unique_modalities):
                _ = self.modality_cache.get(m)

    def __len__(self):
        return len(self.sample_ids)

    def _generate_position_indices(self, n_pd, n_ph, n_pw):
        """
        Create per-patch 3D indices in (d, h, w) order for a regular patch grid.

        Args:
            n_pd (int): Number of patches along depth.
            n_ph (int): Number of patches along height.
            n_pw (int): Number of patches along width.

        Returns:
            torch.Tensor: Tensor of shape (n_patches, 3) with (d, h, w) integer indices,
                flattened in row-major order. Aligns with DHW layout used by patchify_3d.
        """
        d = torch.arange(n_pd, dtype=torch.long)
        h = torch.arange(n_ph, dtype=torch.long)
        w = torch.arange(n_pw, dtype=torch.long)
        D, H, W = torch.meshgrid(d, h, w, indexing="ij")  # (n_pd, n_ph, n_pw)
        return torch.stack([D, H, W], dim=-1).reshape(-1, 3)  # (n_patches, 3)

    def _get_modality_embedding(self, mod_name: str, repeat_n: int) -> torch.Tensor:
        """
        Fetch a cached modality embedding for `mod_name` (stored on CPU),
        and repeat it `repeat_n` times to align with patch tokens.

        Returns:
            torch.Tensor: (repeat_n, EmbDim) on CPU.
        """
        emb = self.modality_cache.get(mod_name)          # (EmbDim,)
        return emb.unsqueeze(0).repeat(repeat_n, 1)      # (N, EmbDim)

    def _process_modality(self, mod_name, volume):
        """
        Patchify a single modality volume and build per-patch metadata.

        Args:
            mod_name (str): Modality name (e.g., 'T1', 'T2', 'FLAIR').
            volume (torch.Tensor): 3D tensor (D, H, W) in DHW order (already CPU float).

        Returns:
            tuple:
                patches (torch.Tensor): (N, PatchDim) flattened patch vectors.
                modality_embedding (torch.Tensor): (N, EmbDim) repeated embedding per patch.
                pos_indices (torch.Tensor): (N, 3) integer (d, h, w) patch indices.
                n_pd (int): Patch grid depth size.
                n_ph (int): Patch grid height size.
                n_pw (int): Patch grid width size.
        """
        # volume shape (D, H, W) - add batch and modality dim for patchify
        volume = volume.unsqueeze(0).unsqueeze(0) # (1, 1, D, H, W)
        patches, patch_info = patchify_3d(volume, self.patch_size)
        (_, n_pd, n_ph, n_pw, *_ ) = patch_info
        # patches shape: (1, n_patches, patch_volume) -> (n_patches, patch_volume)
        patches = patches.squeeze(0)

        modality_embedding = self._get_modality_embedding(mod_name, patches.shape[0])

        pos_indices = self._generate_position_indices(n_pd, n_ph, n_pw)

        return patches, modality_embedding, pos_indices, n_pd, n_ph, n_pw

    def __getitem__(self, idx):
        sample_id  = self.sample_ids[idx]
        modality_dict = self.sample_dict[sample_id]
        modality_volumes = load_modalities(modality_dict)

        all_patches = []
        all_modality_embeddings = []
        all_position_indices = [] # List to store (d, h, w) indices for each patch

        max_d, max_h, max_w = 0, 0, 0 # Track max dimensions for pos embedding

        for mod_name, volume in modality_volumes.items():
            patches, modality_embedding, pos_indices, n_pd, n_ph, n_pw = self._process_modality(mod_name, volume)
            max_d = max(max_d, n_pd)
            max_h = max(max_h, n_ph)
            max_w = max(max_w, n_pw)

            all_patches.append(patches)
            all_modality_embeddings.append(modality_embedding)
            all_position_indices.append(pos_indices)


        # Concatenate patches and embeddings from all modalities for this sample
        sample_patches  = torch.cat(all_patches, dim=0)             # (TotalPatches, PatchDim)
        sample_mod_embs = torch.cat(all_modality_embeddings, dim=0) # (TotalPatches, ModEmbDim)
        sample_pos_idxs = torch.cat(all_position_indices, dim=0)    # (TotalPatches, 3)

        return {
            "patches"            : sample_patches,
            "modality_embeddings": sample_mod_embs,
            "position_indices"   : sample_pos_idxs,
            "patch_grid_size"    : {
                "n_pd": max_d,
                "n_ph": max_h,
                "n_pw": max_w
            } # (for positional embedding)
        }
    
if __name__ == "__main__":
    import tempfile
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        vol1 = np.random.rand(128, 128, 128).astype(np.float32)
        vol2 = np.random.rand(128, 128, 128).astype(np.float32)
        np.save(f"{tmpdir}/t1.npy", vol1.transpose(1, 2, 0))   # (H, W, D)
        np.save(f"{tmpdir}/t2.npy", vol2.transpose(1, 2, 0))

        # Sample dict: one subject, two modalities
        sample_dict = {
            "case001": {
                "T1": f"{tmpdir}/t1.npy",
                "T2": f"{tmpdir}/t2.npy",
            }
        }

        class DummyEncoder:
            def __call__(self, text: str):
                return torch.ones(1, 8) * (len(text))  # (1, EmbDim)

        dataset = MultiModMRIDataset(
            sample_dict=sample_dict,
            modality_encoder=DummyEncoder(),
            patch_size=(16, 16, 16),
            precompute_modalities=True,
            cache_device="cpu",
        )

        sample = dataset[0]
        print("patches:", sample["patches"].shape)
        print("modality_embeddings:", sample["modality_embeddings"].shape)
        print("position_indices:", sample["position_indices"].shape)
        print("patch_grid_size:", sample["patch_grid_size"])