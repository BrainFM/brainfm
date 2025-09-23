import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from brainfm.utils import Config
from brainfm.utils import load_json
from brainfm.models.encoder import build_modality_encoder
from .dataset import MultiModMRIDataset

def mri_collate_fn(batch):
    patches_list   = [item['patches'] for item in batch]
    mod_embs_list  = [item['modality_embeddings'] for item in batch]
    pos_idxs_list  = [item['position_indices'] for item in batch]

    # Pad sequences
    # Need batch_first=True for pad_sequence output shape consistency with transformers
    padded_patches  = pad_sequence(patches_list, batch_first=True, padding_value=0.0)
    padded_mod_embs = pad_sequence(mod_embs_list, batch_first=True, padding_value=0.0)
    padded_pos_idxs = pad_sequence(pos_idxs_list, batch_first=True, padding_value=-1) # Use -1 for padding index

    # Create attention mask for Transformer: True indicates **padding** (to be ignored).
    # Different libraries use different conventions; PyTorch expects True at padded positions.
    lengths = torch.tensor([len(p) for p in patches_list])
    max_len = padded_patches.shape[1]
    pad_mask = torch.arange(max_len)[None, :] >= lengths[:, None] # (B, TotalPatches), True where padded

    return {
        "patches": padded_patches,
        "modality_embeddings": padded_mod_embs,
        "position_indices": padded_pos_idxs,
        "pad_mask": pad_mask,
    }

def build_loader(config: Config, logger=None) -> DataLoader: 
    # Load data mapping dict
    mapping_file = config.paths.mapping_file
    sample_dict = load_json(mapping_file)

    # Initialize the modality embedding
    modality_encoder = build_modality_encoder(config=config, logger=logger)

    # Initialize the dataset
    dataset = MultiModMRIDataset(
        sample_dict=sample_dict,
        modality_encoder=modality_encoder,
        patch_size=config.data.patch_size
    )

    # Create DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        collate_fn=mri_collate_fn,
        pin_memory=config.data.pin_memory,
        num_workers=config.data.num_workers,
        shuffle=True,
    )

    # Log dataset size
    if logger:
        sample = next(iter(data_loader))
        logger.info(f"[Sample Shapes] patches (B, TotalPatches, PatchDim): {tuple(sample['patches'].shape)}, "
                    f"modality_embeddings (B, TotalPatches, ModEmbDim): {tuple(sample['modality_embeddings'].shape)}, "
                    f"position_indices (B, TotalPatches, 3): {tuple(sample['position_indices'].shape)}, "
                    f"pad_mask (B, TotalPatches): {tuple(sample['pad_mask'].shape)}")

    return data_loader


if __name__ == "__main__":
    import os
    import numpy as np
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        modalities = ['T1', 'T2']
        sample_dict = {}
        for i in range(2):
            sample_id = f"sample{i}"
            sample_dict[sample_id] = {}
            for mod in modalities:
                arr = np.random.rand(128, 128, 128).astype(np.float32)
                fpath = os.path.join(tmpdir, f"{sample_id}_{mod}.npy")
                np.save(fpath, arr)
                sample_dict[sample_id][mod] = fpath

        class DummyEncoder:
            def __call__(self, text: str):
                return torch.ones(1, 8) * (len(text))  # (1, EmbDim)

        modality_encoder = DummyEncoder()
        patch_size = (16, 16, 16)

        dataset = MultiModMRIDataset(
            sample_dict=sample_dict,
            modality_encoder=modality_encoder,
            patch_size=patch_size
        )
        loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=mri_collate_fn
        )
        batch = next(iter(loader))
        print("patches shape:", batch["patches"].shape)
        print("modality_embeddings shape:", batch["modality_embeddings"].shape)
        print("position_indices shape:", batch["position_indices"].shape)
        print("pad_mask shape:", batch["pad_mask"].shape)