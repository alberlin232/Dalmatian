import copy
import torch
import torch.nn as nn
from typing import Optional

def _get_clones(module, n):
    """ Clone a module n times. """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

class SPOTERTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    A modified TransformerDecoderLayer that omits the self-attention mechanism, focusing
    solely on cross-attention with the memory.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        # del self.self_attn

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # Use **kwargs to absorb additional unused arguments like 'tgt_is_causal'
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class SPOTER(nn.Module):
    """
    SPOTER - A Transformer-based architecture for sign language recognition from sequences of skeletal data.
    """
    def __init__(self, num_classes, hidden_dim=55):
        super().__init__()

        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim))
        self.pos = nn.Parameter(torch.cat([self.row_embed[0].unsqueeze(0).repeat(1, 1, 1)], dim=-1).flatten(0, 1).unsqueeze(0))
        self.class_query = nn.Parameter(torch.rand(1, hidden_dim))
        self.transformer = nn.Transformer(hidden_dim, 9, 6, 6)
        self.linear_class = nn.Linear(hidden_dim, num_classes)

        # Replace the default decoder layers with custom ones
        custom_decoder_layer = SPOTERTransformerDecoderLayer(hidden_dim, self.transformer.nhead, 2048, 0.1, "relu")
        self.transformer.decoder.layers = _get_clones(custom_decoder_layer, self.transformer.decoder.num_layers)

    def forward(self, inputs):
        h = torch.unsqueeze(inputs.flatten(start_dim=1), 1).float()
        h = self.transformer(self.pos + h, self.class_query.unsqueeze(0)).transpose(0, 1)
        res = self.linear_class(h)

        return res

# The module can be instantiated and used in training or inference as usual
if __name__ == "__main__":
    # Example instantiation and a dummy input processing
    model = SPOTER(num_classes=10, hidden_dim=55)
    dummy_input = torch.rand(1, 50, 55)  # Example input tensor
    output = model(dummy_input)
    print(output)
