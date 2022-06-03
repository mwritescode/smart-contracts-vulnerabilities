from torch import nn

from src.utils.registry import REGISTRY

class Head(nn.Module):
    def __init__(self, in_features=512):
        super(Head, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features//2),      
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features//2, out_features=1)
            )
    
    def forward(self, inputs):
        return self.layers(inputs)

@REGISTRY.register('multitask')
class MultitaskModel(nn.Module):
    def __init__(self, backbone, head, num_heads=5, head_args={}):
        super(MultitaskModel, self).__init__()
        self.backbone = backbone
        self.head_module = head
        self.head_args = head_args
        self.heads = nn.ModuleList()
        self.extended_heads = None
        for _ in range(num_heads):
            self.heads.append(self.head_module(**head_args))
    
    def extend(self, num_heads):
        self.extended_heads = nn.ModuleList()
        for _ in range(num_heads):
            self.extended_heads.append(self.head_module(**self.head_args))
    
    def forward(self, inputs):
        features = self.backbone(inputs)
        outputs = []
        for head in self.heads:
            outputs.append(head(features))
        if self.extended_heads is not None:
            for head in self.extended_heads:
                outputs.append(head(features))
        return outputs
    
    def get_layer_groups(self):
        feature_extractor = [layer for layer in self.backbone.modules()]
        head_layers = [layer for head in self.heads for layer in head.modules()]
        extended_heads = [layer for head in self.extended_heads if self.extended_heads is not None for layer in head.modules()]
        param_groups = {
            'backbone': feature_extractor,
            'heads': head_layers,
            'extended_heads': extended_heads
        }
        return param_groups
