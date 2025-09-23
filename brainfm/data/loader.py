import torch
from torch.utils.data import Dataset, DataLoader
from brainfm.utils import Config
from brainfm.utils import load_json
from brainfm.models.encoder import build_modality_encoder
from .dataset import MultiModMRIDataset

def mri_collate_fn(batch):
    """
    Collate function for batching raw MRI volumes and modality embeddings, padding along the modality dimension.
    Args:
        batch: list of dicts, each with keys:
            - "image": (M, D, H, W)
            - "modality_embs": (M, E)
            - "modality_names": list[str]
    Returns:
        dict with keys:
            - "images": (B, M_max, D, H, W)
            - "modality_embs": (B, M_max, E)
            - "modality_mask": (B, M_max)  # True = PAD
            - "modality_names": list of lists of str
    """
    import torch
    B = len(batch)
    M_max = max(item["image"].shape[0] for item in batch)
    # Check spatial shape
    dhw_set = set(tuple(item["image"].shape[1:]) for item in batch)
    if len(dhw_set) != 1:
        raise ValueError(f"All spatial shapes must match across batch. Got: {dhw_set}")
    D, H, W = batch[0]["image"].shape[1:]
    E = batch[0]["modality_embs"].shape[1]
    images = torch.zeros(B, M_max, D, H, W, dtype=torch.float32)
    modality_embs = torch.zeros(B, M_max, E, dtype=torch.float32)
    modality_mask = torch.ones(B, M_max, dtype=torch.bool)  # True = PAD
    for b, item in enumerate(batch):
        M_b = item["image"].shape[0]
        images[b, :M_b] = item["image"]
        modality_embs[b, :M_b] = item["modality_embs"]
        modality_mask[b, :M_b] = False
    return {
        "images": images,
        "modality_embs": modality_embs,
        "modality_mask": modality_mask,
        "modality_names": [item["modality_names"] for item in batch],
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
        logger.info(f"[Sample Shapes] images (B, M_max, D, H, W): {tuple(sample['images'].shape)}, "
                    f"modality_embs (B, M_max, E): {tuple(sample['modality_embs'].shape)}, "
                    f"modality_mask (B, M_max): {tuple(sample['modality_mask'].shape)}, "
                    f"modality_names: list[list[str]] with len={len(sample['modality_names'])}")

    return data_loader


if __name__ == "__main__":
    import os
    import numpy as np
    import tempfile
    # Build a dataset with two samples, each with two modalities
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
        print("images shape:", batch["images"].shape)
        print("modality_embs shape:", batch["modality_embs"].shape)
        print("modality_mask shape:", batch["modality_mask"].shape)