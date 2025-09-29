import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from monai.transforms import Compose, EnsureType, NormalizeIntensity, DivisiblePad
from brainfm.utils import to_3tuple

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
 
def make_same_shape(vol, target_shape):
    """Crop or pad a 3D volume to match target_shape (D, H, W)."""
    D, H, W = vol.shape
    target_D, target_H, target_W = target_shape
    # Crop or pad depth (D)
    if D > target_D:
        start = (D - target_D) // 2
        vol = vol[start:start+target_D, :, :]
    elif D < target_D:
        pad_before = (target_D - D) // 2
        pad_after = target_D - D - pad_before
        vol = torch.nn.functional.pad(vol, (0, 0, 0, 0, pad_before, pad_after), mode='constant', value=0)
    # Crop or pad height (H)
    if H > target_H:
        start = (H - target_H) // 2
        vol = vol[:, start:start+target_H, :]
    elif H < target_H:
        pad_before = (target_H - H) // 2
        pad_after = target_H - H - pad_before
        vol = torch.nn.functional.pad(vol, (0, 0, pad_before, pad_after), mode='constant', value=0)
    # Crop or pad width (W)
    if W > target_W:
        start = (W - target_W) // 2
        vol = vol[:, :, start:start+target_W]
    elif W < target_W:
        pad_before = (target_W - W) // 2
        pad_after = target_W - W - pad_before
        vol = torch.nn.functional.pad(vol, (pad_before, pad_after, 0, 0, 0, 0), mode='constant', value=0)
    return vol

def load_modalities(modality_paths, img_size=(128,128,128)):
    """Load multiple modality volumes from file paths; keys are modality names.
    Returns a dict of {modality: torch.FloatTensor} with shape (D, H, W) on CPU.
    If img_size is given, each volume is cropped/padded to that shape.
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
        
        # Resize to target shape if specified
        if img_size is not None:
            vol = make_same_shape(vol, target_shape=(128, 128, 128))

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
    Dataset for multi-modality 3D MRI that returns raw stacked volumes and per-modality text embeddings.
    The dataset always normalizes intensities and pads volumes to be divisible by the patch size.
    
    Returns:
      - image: torch.FloatTensor of shape (M, D, H, W) stacked volumes for M modalities.
      - modality_names: list of modality names in the order of stacking.
      - modality_embs: torch.FloatTensor of shape (M, EmbDim) with cached text embeddings.
    """
    def __init__(self,
                 sample_dict,
                 modality_encoder,
                 patch_size,
                 img_size=(128,128,128),
                 precompute_modalities: bool = True,
                 cache_device: str = "cpu"):
        
        validate_sample_dict(sample_dict)

        self.sample_dict = sample_dict
        self.sample_ids  = list(self.sample_dict.keys())
        self.patch_size  = to_3tuple(patch_size)
        self.img_size    = to_3tuple(img_size)

        # Wrap encoder with a cache (embeddings stored on CPU by default)
        self.modality_cache = ModalityEncoderCache(
            modality_encoder,
            cache_device=cache_device
        )

        self.preprocess = Compose([
            EnsureType(),
            NormalizeIntensity(nonzero=True, channel_wise=True),
            DivisiblePad(k=self.patch_size),
        ])

        # Optionally precompute all unique modality names found in the dataset
        if precompute_modalities:
            unique_modalities = set()
            for _sid, mods in self.sample_dict.items():
                unique_modalities.update(list(mods.keys()))
            for m in sorted(unique_modalities):
                _ = self.modality_cache.get(m)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id  = self.sample_ids[idx]
        modality_dict = self.sample_dict[sample_id]
        modality_volumes = load_modalities(modality_dict, img_size=self.img_size) # {modality: (D, H, W) tensor}, all modality volume has same img_shape

        # preserve modality order as in modality_dict keys
        image = torch.stack([modality_volumes[m] for m in modality_dict.keys()], dim=0)  # (M, D, H, W)
        image = self.preprocess(image) # apply MONAI transforms on (C=M, D, H, W)
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        emb_list = [self.modality_cache.get(m) for m in modality_dict.keys()]
        modality_embs = torch.stack(emb_list, dim=0)  # (M, EmbDim)

        return {
            "image": image,
            "modality_names": list(modality_dict.keys()),
            "modality_embs": modality_embs
        }
    
if __name__ == "__main__":
    import tempfile
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        vol1 = np.random.rand(256, 256, 128).astype(np.float32)
        vol2 = np.random.rand(336, 224, 336).astype(np.float32)
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
        print("image:", sample["image"].shape)
        print("modality_embs:", sample["modality_embs"].shape)