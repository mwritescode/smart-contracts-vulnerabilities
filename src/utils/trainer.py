from tqdm import tqdm
from src.utils.train_helpers import TrainStepHelper
class Trainer():
    def __init__(self, model, train_dataloader, val_dataloader=None, device='cuda', train_helper=TrainStepHelper):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.train_helper = train_helper
    
    def compile(self, optimizer, loss, metrics=[]):
        self.optimizer = optimizer
        self.criterion = loss
        self.metrics = metrics
        self.logs = self.__init_logs_dict()
        self.train_helper = self.train_helper(self.model, self.criterion, self.device)
    
    def __init_logs_dict(self):
        logs = {
            'epoch_num': 0, 
            'train_batches_per_epoch': len(self.train_dataloader), 
            'val_batches_per_epoch': len(self.val_dataloader) if self.val_dataloader is not None else None,
            'train': {}, 
            'val': {}, 
            'metrics': {}}
        for step in ['train', 'val']:
            logs[step]['loss'] = 0.0
            logs[step]['predictions'] = []
            logs[step]['labels'] = []
            logs[step]['batch_idx'] = 0
        for metric in self.metrics.keys():
            logs['metrics']['train_'+metric] = 0.0
            logs['metrics']['val_'+metric] = 0.0
        return logs
    
    def fit(self, epochs, callbacks):
        if self.device == 'cuda':
            self.model.cuda()

        for i in range(epochs):
            print(f'Epoch {i}:')
            self.logs = self.__init_logs_dict()
            self.logs['epoch_num'] = i
            self.__train_epoch(callbacks)
            for callback in callbacks:
                self.logs = callback.on_train_epoch_end(self.logs)

            if self.val_dataloader is not None:
                self.__val_epoch(callbacks)
                for callback in callbacks:
                    self.logs = callback.on_val_epoch_end(self.logs)
            
            if 'early_stop' in self.logs.keys():
                break
            
            self.train_helper.reset()
            print('train_loss: {:.4f} | val_loss: {:.4f} |'.format(self.logs['train']['loss'], self.logs['val']['loss']), end=' ')
            print(" | ".join([ '{}: {:.4f}'.format(metric_name, metric_val) for metric_name, metric_val in self.logs['metrics'].items()]), end='\n\n')
            
        for callback in callbacks:
            self.logs = callback.on_train_end(self.logs)
    
    def __train_epoch(self, callbacks):
        self.model.train()
        running_metrics = [0 for _ in self.metrics]
        pbar = tqdm(self.train_dataloader, desc='Training...')
        for data in pbar:
            preds, labels, total_loss, loss = self.train_helper.step(data, mode='train')
            self.__update_loss_and_preds(labels, preds, total_loss, mode='train')

            running_metrics = self.__compute_batch_metrics(running_metrics, labels, preds, mode='train')
            pbar.set_postfix({'loss': self.logs['train']['loss'], **{metric_name: self.logs['metrics']['train_' + metric_name] for metric_name in self.metrics.keys()}})

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            for callback in callbacks:
                self.logs = callback.on_train_batch_end(self.logs)
    
    def __update_loss_and_preds(self, labels, preds, total_loss, mode='train'):
        self.logs[mode]['predictions'] += preds.tolist()
        self.logs[mode]['labels'] += labels.tolist()
        self.logs[mode]['loss'] = total_loss/(self.logs[mode]['batch_idx']+1)
    
    def __compute_batch_metrics(self, running_metrics, labels, preds, mode='train'):
        for i, (metric_name, metric_func) in enumerate(self.metrics.items()):
            running_metrics[i] += metric_func(labels.tolist(), preds.tolist())
            self.logs['metrics'][mode + '_' + metric_name] = running_metrics[i]/(self.logs[mode]['batch_idx']+1)
        self.logs[mode]['batch_idx'] += 1
        return running_metrics
    
    def __val_epoch(self, callbacks):
        self.model.eval()
        total_loss = 0.0
        running_metrics = [0 for _ in self.metrics]
        pbar = tqdm(self.val_dataloader, desc='Validation...')
        for data in pbar:
            preds, labels, total_loss, _ = self.train_helper.step(data, mode='val')
            self.__update_loss_and_preds(labels, preds, total_loss, mode='val')

            running_metrics = self.__compute_batch_metrics(running_metrics, labels, preds, mode='val')
            pbar.set_postfix({'loss': self.logs['val']['loss'], **{metric_name: self.logs['metrics']['val_' + metric_name] for metric_name in self.metrics.keys()}})

            for callback in callbacks:
                self.logs = callback.on_val_batch_end(self.logs)