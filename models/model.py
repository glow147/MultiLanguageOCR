import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinConfig, SwinModel

class TransformerDecoder(nn.Module):
    def __init__(self, target_len, output_dim, d_model, nhead, num_layers, dim_feedforward, dropout, activation):
        super().__init__()

        self.output_dim = output_dim

        # Embedding layer
        self.embedding = nn.Embedding(output_dim, d_model)
        self.pos_embedding = nn.Embedding(target_len, d_model)

        # Transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=num_layers
        )

        # Fully connected layer
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, targets, memory):
        target_embedded = self.embedding(targets) # embedded: [batch size, trg len, d_model]
        positions = torch.arange(0, target_embedded.size(1), device=target_embedded.device).unsqueeze(0).repeat(target_embedded.size(0), 1)
        target_embedded += self.pos_embedding(positions)
        target_embedded = target_embedded.transpose(0, 1) # embedded: [trg len, batch size, d_model]
        target_masks = nn.Transformer.generate_square_subsequent_mask(targets.shape[1]).to(targets.device)

        memory = memory.transpose(0, 1) # memory: [src len, batch size, d_model]
        
        # 디코더를 통과
        output = self.transformer_decoder(target_embedded, memory, target_masks) # output: [trg len, batch size, d_model]
        output = output.transpose(0, 1) # output: [batch size, trg len, d_model]
        output = self.fc_out(output)

        return output


class SwinTransformerEncoder(nn.Module):
    def __init__(self, image_size, patch_size, window_size):
        super(SwinTransformerEncoder, self).__init__()
        # Swin Transformer 모델 구성 변경
        config = SwinConfig(
            num_channels=1,
            depths= [2, 6, 2],
            num_heads= [6, 12, 24],
            image_size=image_size,  # 이미지 크기 변경 (예: [112, 224]로 변경 가능)
            patch_size=patch_size,  # 패치 크기 변경
            window_size=window_size,  # 윈도우 크기 변경
        )
    
        # 변경된 구성으로 모델 초기화
        self.backbone = SwinModel(config)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)


    def forward(self, x):
        out = self.backbone(pixel_values=x).last_hidden_state

        return out
    

