import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from brainfm.utils import Config
from brainfm.utils import load_json
from brainfm.models.encoder import build_modality_encoder
from .dataset import MultiModMRIDataset

def mri_collate_fn(batch, device=None):
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
    attention_mask = torch.arange(max_len)[None, :] >= lengths[:, None] # (B, TotalPatches), True where padded


    return {
        "patches": padded_patches,
        "modality_embeddings": padded_mod_embs,
        "position_indices": padded_pos_idxs,
        "attention_mask": attention_mask,
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

    # # Create DataLoader
    # data_loader = DataLoader(
    #     dataset,
    #     batch_size=config.DATA.BATCH_SIZE,
    #     shuffle=True,
    #     collate_fn=mri_collate_fn,
    #     pin_memory=config.DATA.PIN_MEMORY,
    #     num_workers=config.DATA.NUM_WORKERS,
    # )

    # # Log dataset size
    # if logger:
    #     sample = next(iter(data_loader))
    #     logger.info(f"[Sample Shapes] patches (B, TotalPatches, PatchDim): {tuple(sample['patches'].shape)}, "
    #                 f"modality_embeddings (B, TotalPatches, ModEmbDim): {tuple(sample['modality_embeddings'].shape)}, "
    #                 f"position_indices (B, TotalPatches, 3): {tuple(sample['position_indices'].shape)}, "
    #                 f"attention_mask (B, TotalPatches): {tuple(sample['attention_mask'].shape)}")

    # return data_loader