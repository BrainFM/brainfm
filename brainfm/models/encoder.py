import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Avoid warnings from the tokenizers library
# when using multiprocessing
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ModalityEncoder(nn.Module):
    def __init__(self, model_name="dmis-lab/biobert-v1.1"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model_name = model_name

        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()

    @torch.inference_mode()
    def forward(self, modality_name):
        """
        Encodes a modality name string into a fixed embedding vector.
        Uses mean pooling of the last hidden state.
        """
        if not isinstance(modality_name, str):
             raise TypeError(f"Expected string modality name, got {type(modality_name)}")

        inputs = self.tokenizer(modality_name, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v for k, v in inputs.items()}

        outputs = self.model(**inputs)

        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask'] # (batch_size, seq_len)

        # Perform mean pooling - mask out padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9) # Avoid division by zero
        mean_pooled_embedding = sum_embeddings / sum_mask

        # (batch_size=1, hidden_dim) -> (hidden_dim,)
        return mean_pooled_embedding.squeeze(0)

    def get_embedding_dim(self):
        # Helper to get the output dimension
        return self.model.config.hidden_size

    def _get_device(self):
        # Helper to get device of model parameters
        return next(self.model.parameters()).device


def build_modality_encoder(config, logger=None):
    # Load the modality embedding model
    model_name = config.data.modality_encoder_model

    if logger:
        logger.info(
            f"Using HuggingFace model for the modality embedding: {model_name}"
        )
    return ModalityEncoder(model_name=model_name)

if __name__ == "__main__":
    biobert = ModalityEncoder()
    t1_emb = biobert("t1")
    print(t1_emb.shape) # Should be (768,) for BioBERT v1.1
    print(biobert.get_embedding_dim()) # Should be 768