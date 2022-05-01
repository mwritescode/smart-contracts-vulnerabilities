from src.utils.registry import REGISTRY

@REGISTRY.register('resnet_train_helper')
class TrainStepHelper:
    def __init__(self, model, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.total_loss = {
            'train': 0.0,
            'val': 0.0
        }
    
    def step(self, data, mode='train'):
        images = data[0].to(self.device)
        labels = data[1].to(self.device)

        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        self.total_loss[mode] += loss.item()
        preds = (outputs >= 0.0).float()
        return preds, labels, self.total_loss[mode], loss
    
    def reset(self):
        self.total_loss = {
            'train': 0.0,
            'val': 0.0
        }

@REGISTRY.register('inception_train_helper')    
class InceptionTrainHelper(TrainStepHelper):
    def __init__(self, model, criterion, device):
        super().__init__(model, criterion, device)
    
    def step(self, data, mode='train'):
        images = data[0].to(self.device)
        labels = data[1].to(self.device)

        if mode == 'train':
            outputs, aux_outputs = self.model(images)
            loss1 = self.criterion(outputs, labels)
            loss2 = self.criterion(aux_outputs, labels)
            loss = loss1 + 0.4*loss2
        else: 
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

        self.total_loss[mode] += loss.item()
        preds = (outputs >= 0.0).float()
        return preds, labels, self.total_loss[mode], loss