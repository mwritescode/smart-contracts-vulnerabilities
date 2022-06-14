import torch
from torch import nn

from src.utils.registry import REGISTRY

@REGISTRY.register('rnn')
class LSTMNetwork(nn.Module):
    def __init__(self, num_classes=1, classify=True, vocabulary_size=257):
        super(LSTMNetwork, self).__init__()
        self.layers = 3
        self.hidden_size = 128
        self.classify = classify
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=150, padding_idx=256)
        self.dropout1 = nn.Dropout(0.1)
        self.lstm = nn.LSTM(bidirectional=True, input_size=150, hidden_size=self.hidden_size, batch_first=True, num_layers=self.layers)
        self.dense1 = nn.Linear(in_features=256, out_features=512)
        self.dropout2 = nn.Dropout(0.1)
        if classify:
            self.dense2 = nn.Linear(in_features=512, out_features=num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        lenghts = inputs.shape[1] - (inputs == 256).sum(dim=1).to('cpu')
        out = self.dropout1(self.embedding(inputs))

        out = nn.utils.rnn.pack_padded_sequence(out, lenghts, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(out)
        h_n = h_n.view(self.layers, 2, inputs.shape[0], self.hidden_size)
        last_hidden = h_n[-1]
        last_hidden_fwd = last_hidden[0]
        last_hidden_bwd = last_hidden[1]
        out = torch.cat((last_hidden_fwd, last_hidden_bwd), 1)

        out = self.dropout2(self.relu(self.dense1(out)))
        if self.classify:
            out = self.dense2(out)
        return out

    def get_layer_groups(self):
        linear_layers = [elem[1] for elem in self.dense2.named_parameters()]
        other_layers = [elem[1] for elem in filter(lambda param_tuple: 'dense2' not in param_tuple[0], self.named_parameters())]
        param_groups = {
            'classifier': linear_layers,
            'feature_extractor': other_layers 
        }
        return param_groups