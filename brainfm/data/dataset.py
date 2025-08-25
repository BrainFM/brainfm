import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from brainfm.utils import to_3tuple
from brainfm.models import patchify_3d

def load_nifti(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return nib.load(path).get_fdata()


def load_npy(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(path).transpose(2, 0, 1)  # (H, W, D) → (D, H, W)


def load_mri(modality_dict, dummy=False):
    volumes = {}

    for modality, path in modality_dict.items():
        if dummy:
            volumes[modality] = np.random.randn(128, 128, 128).astype(np.float32)
        else:
            # Load the actual MRI data
            if path.endswith('.nii.gz'):
                volumes[modality] = load_nifti(path)
            elif path.endswith(".npy"):
                volumes[modality] = load_npy(path)
            else:
                raise ValueError(f"Unsupported file format for {modality}: {path}")

    return volumes

class MultiModMRIDataset(Dataset):
    def __init__(self, sample_dict, modality_encoder, patch_size):
        """
        sample_dict: Dict mapping sample_id to {
            'modality_name': path_to_modality_file
        }
        modality_encoder: Initialized modality embedding
        patch_size: Tuple (p_d, p_h, p_w)
        """
        
        #validate_sample_dict(sample_dict)

        self.sample_dict      = sample_dict
        self.sample_ids       = list(self.sample_dict.keys())
        self.modality_encoder = modality_encoder
        self.patch_size       = to_3tuple(patch_size)
        self.device           = modality_encoder._get_device()

    def __len__(self):
        return len(self.sample_ids)

    def _generate_position_indices(self, n_pd, n_ph, n_pw):
        pos_indices = []
        for d_i in range(n_pd):
            for h_i in range(n_ph):
                for w_i in range(n_pw):
                    pos_indices.append(torch.tensor([d_i, h_i, w_i], dtype=torch.long))
        return torch.stack(pos_indices)  # (n_patches, 3)

    def _process_modality(self, mod_name, volume):
        # volume shape (D, H, W) - add batch and modality dim for patchify
        volume = volume.unsqueeze(0).unsqueeze(0) # (1, 1, D, H, W)
        D, H, W = volume.shape[2:]
        patches, patch_info = patchify_3d(volume, self.patch_size)
        (M, n_pd, n_ph, n_pw, p_d, p_h, p_w) = patch_info
        
        # patches shape: (1, n_patches, patch_volume) -> (n_patches, patch_volume)
        patches = patches.squeeze(0)

        modality_embedding = self.modality_encoder(mod_name) # Get embedding
        modality_embedding = modality_embedding.unsqueeze(0).repeat(patches.shape[0], 1).to(self.device)

        pos_indices = self._generate_position_indices(n_pd, n_ph, n_pw)

        return patches, modality_embedding, pos_indices, n_pd, n_ph, n_pw

    def __getitem__(self, idx):
        sample_id  = self.sample_ids[idx]
        modality_dict = self.sample_dict[sample_id]

        # Load the volumes and convert to tensors
        volumes = {}
        raw_volumes = load_mri(modality_dict, dummy=True)
        for mod_name, np_vol in raw_volumes.items():
            volumes[mod_name] = torch.from_numpy(np_vol).float() if not isinstance(np_vol, torch.Tensor) else np_vol

        all_patches = []
        all_modality_embeddings = []
        all_position_indices = [] # List to store (d, h, w) indices for each patch

        max_d, max_h, max_w = 0, 0, 0 # Track max dimensions for pos embedding

        for mod_name, volume in volumes.items():
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
            "patches": sample_patches,
            "modality_embeddings": sample_mod_embs,
            "position_indices": sample_pos_idxs,
            "max_patch_dim": {"max_d": max_d, "max_h": max_h, "max_w": max_w} # (for positional embedding)
        }