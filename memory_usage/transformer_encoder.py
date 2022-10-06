import torch.nn as nn
transformer_model = nn.Transformer(nhead=1, num_encoder_layers=1, num_decoder_layers=1)

print(transformer_model)
