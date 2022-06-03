from torch import nn
from torchvision import models

from src.utils.registry import REGISTRY

@REGISTRY.register('resnet')
class ResNetModel(nn.Module):
    def __init__(self, num_classes=5, classify=True):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        
        if classify:
            self.resnet.fc = nn.Linear(512, num_classes)
        else:
            features = nn.ModuleList(self.resnet.children())[:-1]
            self.resnet = nn.Sequential(*features).append(nn.Flatten())
    
    def forward(self, inputs):
        return self.resnet(inputs)
    
    def get_layer_groups(self):
        linear_layers = [elem[1] for elem in filter(lambda param_tuple: 'fc' in param_tuple[0], self.resnet.named_parameters())]
        other_layers = [elem[1] for elem in filter(lambda param_tuple: 'fc' not in param_tuple[0], self.resnet.named_parameters())]
        param_groups = {
            'classifier': linear_layers,
            'feature_extractor': other_layers 
        }
        return param_groups