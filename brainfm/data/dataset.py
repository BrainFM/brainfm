import os
from typing import Tuple
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    EnsureType,
    NormalizeIntensity,
    DivisiblePad,
    RandBiasField,
    RandGaussianNoise,
    RandAdjustContrast,
    RandFlip,
    RandAffine
)
from ..utils import to_3tuple

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
    """Load a NIfTI file (.nii.gz) and return a NumPy array (vendor shape) as float32."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    img = nib.load(path)
    return img.get_fdata(dtype=np.float32)


def load_npy_dhw(path):
    """Load a .npy volume saved as (H, W, D) and return as (D, H, W)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(path).transpose(2, 0, 1)  # (H, W, D) → (D, H, W)
 

def make_same_shape(vol: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
    """
    Center-crop then pad a 3D tensor to target (D, H, W).
    Uses a single F.pad call with a 6-tuple: (Wl, Wr, Hl, Hr, Dl, Dr).
    """
    assert vol.ndim == 3, f"Expected (D,H,W), got {tuple(vol.shape)}"
    tD, tH, tW = map(int, target_shape)
    D, H, W = vol.shape

    # 1) Center-crop if larger
    if D > tD:
        s = (D - tD) // 2
        vol = vol[s:s + tD, :, :]
    if H > tH:
        s = (H - tH) // 2
        vol = vol[:, s:s + tH, :]
    if W > tW:
        s = (W - tW) // 2
        vol = vol[:, :, s:s + tW]

    # Update shape after cropping
    D, H, W = vol.shape

    # 2) Center-pad if smaller (build symmetric pads; put the +1 on the "after" side)
    pd = max(tD - D, 0)
    ph = max(tH - H, 0)
    pw = max(tW - W, 0)

    if pd or ph or pw:
        pd0 = pd // 2; pd1 = pd - pd0
        ph0 = ph // 2; ph1 = ph - ph0
        pw0 = pw // 2; pw1 = pw - pw0
        # pad = (W_left, W_right, H_left, H_right, D_left, D_right)
        vol = F.pad(vol, (pw0, pw1, ph0, ph1, pd0, pd1), mode="constant", value=0)

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
            vol = make_same_shape(vol, target_shape=to_3tuple(img_size))

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
                raise ValueError(
                    f"Modality name must be a string, got {type(mod_name)} for sample_id '{sample_id}'."
                )
            if not isinstance(file_path, str):
                raise ValueError(
                    f"File path for modality '{mod_name}' in sample_id '{sample_id}' must be a string, got {type(file_path)}."
                )
            if not os.path.isfile(file_path):
                raise ValueError(
                    f"File path does not exist for modality '{mod_name}' in sample_id '{sample_id}': {file_path}"
                )
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
            # patch_size is DHW order, k expects HWD order)
            DivisiblePad(k=(self.patch_size[1], self.patch_size[2], self.patch_size[0])),
            # ==== Strong 3D augmentations (expects (C, H, W, D)) ====
            RandBiasField(prob=0.3, coeff_range=(0.3, 0.6)),
            RandGaussianNoise(prob=0.3, mean=0.0, std=0.05),
            RandAdjustContrast(prob=0.3, gamma=(0.7, 1.5)),
            RandFlip(prob=0.5, spatial_axis=[0, 1, 2]),
            # ~15 degree rotations around each axis (radians)
            RandAffine(
                prob=0.3,
                rotate_range=(np.pi/12, np.pi/12, np.pi/12),
                translate_range=None,
                scale_range=None,
                padding_mode="border",
            ),
            NormalizeIntensity(nonzero=True, channel_wise=True),
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

    def _load_images(self, modality_dict):
        """Load volumes per modality and stack to (M, D, H, W)."""
        vols = load_modalities(modality_dict, img_size=self.img_size)
        images = torch.stack([vols[m] for m in modality_dict.keys()], dim=0)
        return images  # (M, D, H, W)

    def _apply_transforms(self, images: torch.Tensor) -> torch.Tensor:
        """Apply MONAI transforms: expects (M, D, H, W) -> returns (M, D, H, W)."""
        images_chwd = images.permute(0, 2, 3, 1)  # (M, H, W, D)
        images_chwd = self.preprocess(images_chwd)
        images = images_chwd.permute(0, 3, 1, 2)  # (M, D, H, W)
        return images

    def _sanitize(self, images: torch.Tensor, sample_id: str) -> torch.Tensor:
        """Replace NaNs, clamp outliers; log once per sample if needed."""
        bad = ~torch.isfinite(images)
        if bad.any():
            nbad = int(bad.sum().item())
            print(f"[warn] Non-finite voxels in {sample_id}: {nbad}")
            images = torch.nan_to_num(images, nan=0.0)
            images = torch.clamp(images, -4.0, 4.0)
        return images

    def _get_modality_embeddings(self, modality_dict) -> torch.Tensor:
        """Return (M, EmbDim) in the same order as modality_dict keys."""
        emb_list = [self.modality_cache.get(m) for m in modality_dict.keys()]
        return torch.stack(emb_list, dim=0)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        modality_dict = self.sample_dict[sample_id]

        images = self._load_images(modality_dict)           # (M, D, H, W)
        images = self._apply_transforms(images)             # (M, D, H, W)
        images = self._sanitize(images, sample_id)          # (M, D, H, W)

        modality_embs = self._get_modality_embeddings(modality_dict)  # (M, EmbDim)

        M, D, H, W = images.shape
        assert (D, H, W) == self.img_size, f"Got {(D,H,W)} vs {self.img_size}"
        assert torch.isfinite(images).all(), "Non-finite values in images"

        return {
            "images": images,                         # (M, D, H, W)
            "modality_names": list(modality_dict.keys()),
            "modality_embs": modality_embs,          # (M, EmbDim)
        }
    
if __name__ == "__main__":
    import tempfile
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        vol1 = np.random.rand(256, 256, 128).astype(np.float32)
        vol2 = np.random.rand(336, 224, 336).astype(np.float32)
        np.save(f"{tmpdir}/t1.npy", vol1)   # already (H, W, D)
        np.save(f"{tmpdir}/t2.npy", vol2)

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
        print("image:", sample["images"].shape)
        print("modality_embs:", sample["modality_embs"].shape)

        x = sample["images"]
        assert x.ndim == 4 and x.shape[0] == 2
        assert x.shape[1:] == (128,128,128)  # (D,H,W)
        assert torch.isfinite(x).all()
        assert sample["modality_embs"].shape == (2, 8)