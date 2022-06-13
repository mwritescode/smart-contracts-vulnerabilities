from torch import nn

from src.utils.registry import REGISTRY

@REGISTRY.register('rnn')
class LSTMNetwork(nn.Module):
    def __init__(self, num_classes=1, classify=True, vocabulary_size=257):
        super(LSTMNetwork, self).__init__()
        self.classify = classify
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=100, padding_idx=256)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm = nn.LSTM(bidirectional=True, input_size=100, hidden_size=128, batch_first=True)
        self.dense1 = nn.Linear(in_features=256, out_features=512)
        self.dropout2 = nn.Dropout(0.2)
        if classify:
            self.dense2 = nn.Linear(in_features=512, out_features=num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        out = self.dropout1(self.embedding(inputs))
        out = self.lstm(out)[0][:, -1, :]
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