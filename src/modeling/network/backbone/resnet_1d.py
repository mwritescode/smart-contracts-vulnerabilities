from torch import nn

from src.utils.registry import REGISTRY

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.relu = nn.LeakyReLU()
    
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        out = x + inputs
        return self.relu(out)

@REGISTRY.register('1d_resnet')   
class ResNet1D(nn.Module):
    def __init__(self, num_classes=1, classify=True):
        super(ResNet1D, self).__init__()
        self.classify = classify
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, stride=3),
            nn.BatchNorm1d(num_features=128),
            nn.LeakyReLU(),
            ResBlock1D(in_channels=128, out_channels=128),
            nn.MaxPool1d(kernel_size=3, stride=3),
            ResBlock1D(in_channels=128, out_channels=128),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(),
            ResBlock1D(in_channels=256, out_channels=256),
            nn.MaxPool1d(kernel_size=3, stride=3),
            #ResBlock1D(in_channels=256, out_channels=256),
            #nn.MaxPool1d(kernel_size=3, stride=3),
            ResBlock1D(in_channels=256, out_channels=256),
            nn.MaxPool1d(kernel_size=3, stride=3),
            ResBlock1D(in_channels=256, out_channels=256),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(),
            ResBlock1D(in_channels=512, out_channels=512),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Dropout(0.4),
            nn.Flatten()
        )

        if classify:
            self.classifier = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, inputs):
        out = self.layers(inputs)
        if self.classify:
            out = self.classifier(out)
        return out
    
    def get_layer_groups(self):
        param_groups = {
            'classifier': [elem[1] for elem in self.classifier.named_parameters()],
            'feature_extractor': [elem[1] for elem in self.layers.named_parameters()]
        }
        return param_groups