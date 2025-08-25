import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from .layer import ConditionalLayerNorm

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


# TODO: Replace TransformerEncoderLayer with 
# 3D Swin Blocks using ConditionalLayerNorm in future research.
class ModalityAdaptedTransformerBlock(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation,
                 layer_norm_eps, batch_first, norm_first, cond_dim):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first)

        # Replace standard LayerNorms with ConditionalLayerNorm
        # Assuming norm_first=True (common in modern transformers)
        if norm_first:
            # Keep reference to original norms if needed, but we override them
            self.norm1 = ConditionalLayerNorm(d_model, cond_dim, eps=layer_norm_eps)
            self.norm2 = ConditionalLayerNorm(d_model, cond_dim, eps=layer_norm_eps)
        else:
             # If norm_first=False, the norms are applied after attention/FFN
             # This setup is less common now but handle if necessary
             # self.norm1 = ConditionalLayerNorm(...) # after self_attn
             # self.norm2 = ConditionalLayerNorm(...) # after linear2
             raise NotImplementedError("norm_first=False with CLN has not been implemented")


    def forward(self, src, src_mask=None, src_key_padding_mask=None, cond=None, **kwargs):
        """
        Modified forward to accept and pass conditioning tensor `cond`.
        cond: (B, SeqLen, CondDim)
        """
        if cond is None:
            raise ValueError("Conditional input `cond` is required for ModalityAdaptedTransformerBlock")

        # --- Logic assuming norm_first=True ---
        # Apply norm1 conditionally, then self-attention
        normed_src = self.norm1(src, cond)
        sa_out     = self.self_attn(normed_src, normed_src, normed_src,
                                attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask,
                                need_weights=False)[0] # Get output, ignore weights
        sa_out     = self.dropout1(sa_out)
        src        = src + sa_out # Residual connection

        # Apply norm2 conditionally, then feedforward network
        normed_src2 = self.norm2(src, cond)
        ffn_out     = self.linear2(self.dropout(self.activation(self.linear1(normed_src2))))
        ffn_out     = self.dropout2(ffn_out)
        src         = src + ffn_out # Residual connection
        # --- End norm_first=True logic ---

        return src


class ModalityAdaptedEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout,
                 cond_dim, layer_norm_eps=1e-5, batch_first=True, norm_first=True):
        super().__init__()
        self.layers = nn.ModuleList([
            ModalityAdaptedTransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=nn.GELU(),
                layer_norm_eps=layer_norm_eps,
                batch_first=batch_first,
                norm_first=norm_first,
                cond_dim=cond_dim
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        # Final norm? Optional, sometimes placed after encoder stack
        # self.norm = ConditionalLayerNorm(d_model, cond_dim, eps=layer_norm_eps)

    def forward(self, src, mask=None, src_key_padding_mask=None, cond=None):
        output = src
        for mod in self.layers:
            output = mod(output,
                         src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask,
                         cond=cond)

        # if self.norm is not None:
        #     output = self.norm(output, cond)
        return output
    

if __name__ == "__main__":
    biobert = ModalityEncoder()
    t1_emb = biobert("t1")
    print(t1_emb.shape) # Should be (768,) for BioBERT v1.1
    print(biobert.get_embedding_dim()) # Should be 768

    # Configuration for test
    B = 2           # Batch size
    SeqLen = 10     # Number of tokens per sample
    d_model = 32    # Token embedding dimension
    cond_dim = 16   # Conditioning embedding dimension
    num_layers = 2  # Number of transformer layers
    nhead = 4       # Number of attention heads
    dim_feedforward = 64
    dropout = 0.1

    # Create a dummy ModalityAdaptedEncoder
    encoder = ModalityAdaptedEncoder(
        num_layers=num_layers, 
        d_model=d_model, 
        nhead=nhead, 
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        cond_dim=cond_dim,
        layer_norm_eps=1e-5, 
        batch_first=True, 
        norm_first=True
    )
    
    # Create dummy input tensors:
    # src: (B, SeqLen, d_model)
    src = torch.randn(B, SeqLen, d_model)
    # cond: (B, SeqLen, cond_dim)
    cond = torch.randn(B, SeqLen, cond_dim)
    
    # Optionally, create a dummy attention mask if needed (here, None)
    mask = None
    src_key_padding_mask = None

    # Run the forward pass
    output = encoder(
        src,
        mask=mask,
        src_key_padding_mask=src_key_padding_mask,
        cond=cond
    )
    
    # Print shapes to verify
    print("Input shape:", src.shape)            # Expected: (B, SeqLen, d_model)
    print("Condition shape:", cond.shape)       # Expected: (B, SeqLen, cond_dim)
    print("Output shape:", output.shape)        # Expected: (B, SeqLen, d_model)
