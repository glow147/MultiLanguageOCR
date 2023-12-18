import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, ResNet18_Weights

import torch.nn.functional as F
from transformers import SwinConfig, SwinModel


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch size, dec_hid_dim]
        # encoder_outputs: [batch size, src_len, enc_hid_dim]

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # 현재 hidden state를 src_len만큼 반복
        hidden = hidden[-1, :, :]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
  
        # encoder_outputs과 hidden states를 결합
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # v를 사용하여 energy의 각 토큰에 대한 하나의 값을 얻음
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)

class GRUAttentionDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, num_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.num_layers = num_layers
        self.attention = Attention(enc_hid_dim, dec_hid_dim)

        # Embedding layer
        self.embedding = nn.Embedding(output_dim, emb_dim)

        # GRU layer
        self.gru = nn.GRU(enc_hid_dim + emb_dim, dec_hid_dim, batch_first=True, bidirectional=True, num_layers=num_layers)

        # Fully connected layer
        self.fc_out = nn.Linear(enc_hid_dim + dec_hid_dim * 2 + emb_dim, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input: [batch size]
        # hidden: [batch size, dec_hid_dim]
        # encoder_outputs: [batch size, enc_hid_dim]

        input = input.unsqueeze(1) # input: [batch size, 1]

        embedded = self.dropout(self.embedding(input)) # embedded: [batch size, 1, emb_dim]

        # Attention score 계산
        # encoder_outputs를 [batch size, 1, enc_hid_dim]로 차원 변경
        if hidden.size(0) != self.num_layers * 2:
            hidden = hidden.repeat(self.num_layers * 2, 1, 1)
        encoder_outputs = encoder_outputs.unsqueeze(1)
        # encoder_outputs = encoder_outputs.repeat(1,2,1)
        a = self.attention(hidden, encoder_outputs) # a: [batch size, 1]

        # 인코더의 출력에 가중치를 곱함
        weighted = a.unsqueeze(2) * encoder_outputs # weighted: [batch size, 1, enc_hid_dim]

        # GRU 입력을 위해 임베딩과 가중 평균을 결합
        gru_input = torch.cat((embedded, weighted), dim=2) # gru_input: [batch size, 1, emb_dim + enc_hid_dim]
       
        output, hidden = self.gru(gru_input, hidden) # output: [batch size, 1, dec_hid_dim]

        # assert (output.squeeze(1) == hidden).all()

        output = output.contiguous().view(output.size(0), -1)
        weighted = weighted.squeeze(1)
        embedded = embedded.squeeze(1)

            
        # 최종 출력 계산
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        return prediction, hidden

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, d_model, nhead, num_layers, dim_feedforward, dropout, activation):
        super().__init__()

        self.output_dim = output_dim
        target_len = output_dim

        # Embedding layer
        self.embedding = nn.Embedding(output_dim, d_model)

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
        out = self.backbone(pixel_values=x).pooler_output

        out = out.unsqueeze(2) # out: [batch size, hidden size, 1]
        return out

class ResnetEncoder(nn.Module):
    def __init__(self):
        super(ResnetEncoder, self).__init__()
        self.backbone = nn.Sequential(*list(resnet18(weights=ResNet18_Weights.DEFAULT).children())[:-1])

    def forward(self, x):
        out = self.backbone(x)

        # out: [batch_size, feature_len]
        out = out.flatten(2)
        return out
    

