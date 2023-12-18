import torch
import torch.nn as nn
import torch.optim as optim

from typing import Any, Dict
from collections import defaultdict
from lightning.pytorch import LightningModule
from models.model import ResnetEncoder, TransformerDecoder

class OcrModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.automatic_optimization = False

        self.lang_dict = cfg['lang_dict']
        self.vocab = cfg['vocab']

        self.feature_extractor = ResnetEncoder()
        self.models_dict = {}

        for language in self.vocab.keys():
            model = TransformerDecoder(output_dim=len(self.vocab[language]['token2id']), **cfg['MODEL_PARAMS'], activation='gelu')
            self.models_dict[language] = model
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    @torch.no_grad()
    def forward(self, image, language):
        image_feature = self.feature_extractor(image)
        vocab = self.vocab[language]
        input_tokens = torch.ones(1, 1).fill_(vocab['token2id']['SOS']).long().to(image_feature.device)
        word = ''
        for i in range(vocab['max_length']):
            out = self.models_dict[language](input_tokens, image_feature.transpose(1,2))
            prob = out[:, -1]
            _, next_word = torch.max(prob, dim=1)
            
            next_word = next_word.item()
            input_tokens = torch.cat([input_tokens,
                            torch.ones(1, 1).long().fill_(next_word).to(image_feature.device)], dim=1)
            if next_word == vocab['token2id']['EOS']:
                break
            else:
                word += vocab['id2token'][str(next_word)]

        return word

    def decoder_eval(self):
        for language in self.models_dict:
            self.models_dict[language].eval()

    def on_train_start(self):
        self.feature_extractor.train()

        for language in self.models_dict:
            self.models_dict[language].to(next(self.feature_extractor.parameters()).device)
            self.models_dict[language].train()

    def training_step(self, batch, batch_idx):
        models_optim = self.optimizers() # [feature_extractor_optim, lang1_optim, lang2_optim, ...]

        models_optim[0].zero_grad()
        
        images, input_tokens, labels, languages = batch
        image_features = self.feature_extractor(images)
        total_loss = 0
        loss_dict = defaultdict(lambda : [])
        for language in self.models_dict:
            language_num = self.lang_dict['token2id'][language]
            models_optim[language_num].zero_grad()
            language_mask = languages == language_num
            if torch.sum(language_mask) == 0:
                continue
            pred = self.models_dict[language](input_tokens[language_mask,], image_features[language_mask,].transpose(1,2))
            loss = self.criterion(pred.view(-1, pred.shape[-1]), labels[language_mask, ].view(-1))
            self.manual_backward(loss, retain_graph=True)
            loss_dict[language] = loss.item()
            total_loss += loss
            models_optim[language_num].step()

        models_optim[0].step()
        self.log_dict(loss_dict, prog_bar=True, on_step=True)

        return {'loss' : total_loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, input_tokens, labels, languages = batch
        image_features = self.feature_extractor(images)
        total_loss = 0
        loss_dict = defaultdict(lambda : [])
        for language in self.models_dict:
            language_num = self.lang_dict['token2id'][language]
            language_mask = languages == language_num
            if torch.sum(language_mask) == 0:
                continue
            pred = self.models_dict[language](input_tokens[language_mask,], image_features[language_mask,].transpose(1,2))
            loss = self.criterion(pred.view(-1, pred.shape[-1]), labels[language_mask, ].view(-1))
            loss_dict[language] = loss.item()
            total_loss += loss
        
        self.log('val_loss', total_loss, prog_bar=True, sync_dist=True)
        return {'val_loss' : total_loss}

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        for language in self.models_dict:
            if language in checkpoint:
                model_state = checkpoint[language]
                self.models_dict[language].load_state_dict(model_state)

        super().on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        for language in self.models_dict:
            checkpoint[language] = self.models_dict[language].state_dict()
        return super().on_save_checkpoint(checkpoint)

    def configure_optimizers(self):
        models_optim = [0] * (len(self.models_dict) + 1) 
        models_optim[0] = optim.AdamW(self.feature_extractor.parameters(), lr=5e-6)

        for language in self.models_dict:
            models_optim[self.lang_dict['token2id'][language]] = optim.AdamW(self.models_dict[language].parameters(), lr=1e-4)

        return models_optim