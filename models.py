import random
import numpy as np
import torch
import torch.nn as nn
from transformers import WavLMModel

class AcousticModel(nn.Module):
    def __init__(self):
        super(AcousticModel, self).__init__()
        self.basemodel = WavLMModel.from_pretrained('microsoft/wavlm-base')
        self.linear = nn.Linear(768, 5)

    def forward(self, x):
        feats = self.basemodel.feature_extractor(x)
        feats = feats.transpose(1, 2)
        feats, _ = self.basemodel.feature_projection(feats)
        emb = self.basemodel.encoder(feats, return_dict=True)[0]
        out = self.linear(emb)

        return emb, out


class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()

        self.hidden_size = 512
        self.blstm = nn.LSTM(input_size=768 * 2,
                             hidden_size=self.hidden_size,
                             num_layers=1,
                             batch_first=True,
                             bidirectional=True)
        self.fc = nn.Linear(self.hidden_size*2, 5)

    def forward(self, x_bert, x_w2v2):

        x_cat = torch.cat((x_bert, x_w2v2), dim=-1)
        x_cat, _ = self.blstm(x_cat)

        out = self.fc(x_cat)

        return out
