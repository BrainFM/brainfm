import torch
import torch.nn as nn

class MAEDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=nn.GELU(), batch_first=True, norm_first=True
        )
        self.layers = nn.TransformerDecoder(decoder_layer, num_layers=num_layers,
                                             norm=nn.LayerNorm(d_model)) # Final norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        tgt: Target sequence (e.g., mask tokens + conditionings)
        memory: Memory sequence (output from encoder for unmasked tokens)
        *_key_padding_mask: Indicate padding tokens
        """
        return self.layers(tgt, memory,
                           tgt_mask=tgt_mask, memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)