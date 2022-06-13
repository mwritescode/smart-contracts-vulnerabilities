from torch import nn
from torchvision import models

from src.utils.registry import REGISTRY

@REGISTRY.register('inception')
class InceptionModel(nn.Module):
    def __init__(self, num_classes=5):
        super(InceptionModel, self).__init__()
        self.inception = models.inception_v3(pretrained=True)
        self.inception.AuxLogits.fc = nn.Linear(768, num_classes)
        self.inception.fc = nn.Linear(2048, num_classes)
    
    def forward(self, inputs):
        return self.inception(inputs)
    
    def get_layer_groups(self):
        linear_layers = [elem[1] for elem in filter(lambda param_tuple: 'fc' in param_tuple[0], self.inception.named_parameters())]
        other_layers = [elem[1] for elem in filter(lambda param_tuple: 'fc' not in param_tuple[0], self.inception.named_parameters())]
        param_groups = {
            'classifier': linear_layers,
            'feature_extractor': other_layers 
        }
        return param_groups