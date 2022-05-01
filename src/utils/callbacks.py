from torch.utils.tensorboard import SummaryWriter
import os
import torch
import copy

from abc import ABC

class Callback(ABC):
    
    def on_train_batch_end(self, logs={}):
        return logs
    
    def on_val_batch_end(self, logs={}):
        return logs
    
    def on_train_epoch_end(self, logs={}):
        return logs
    
    def on_val_epoch_end(self, logs={}):
        return logs
    
    def on_train_end(self, logs={}):
        return logs


class MetricsCallback(Callback):
    def __init__(self, metrics):
        super(Callback, self).__init__()
        self.metric_list = metrics
        self.metric_vals = {}
    
    def __update_logs(self, logs, mode='train'):
        for metric_name, metric_func in self.metric_list.items():
            self.metric_vals[mode + '_' + metric_name] = metric_func(logs[mode]['labels'], logs[mode]['predictions'])
        logs['metrics'].update(self.metric_vals)
        return logs
    
    def on_val_epoch_end(self, logs={}):
        logs = self.__update_logs(logs, mode='val')
        return logs
    
    def on_train_epoch_end(self, logs={}):
        logs = self.__update_logs(logs, mode='train')
        return logs

class TensorBoardLogger(Callback):
    def __init__(self, track_epochwise=[], track_batchwise=[], run_tag=''):
        self.track_epochwise = track_epochwise
        self.track_batchwise = track_batchwise
        run_logs_dir = os.path.join('logs', run_tag)
        self.writers = {
            mode: SummaryWriter(os.path.join(run_logs_dir, mode), flush_secs=1) for mode in ['train', 'val']
        }
    
    def __log_on_tensorboard(self, metric_name, logs, epoch, mode='train'):
        name_with_mode = mode + '_' + metric_name
        metric_val = logs[mode][metric_name] if metric_name == 'loss' else logs['metrics'][name_with_mode]
        #writer.add_scalars(metric_name, {mode: metric_val}, epoch)
        self.writers[mode].add_scalar(metric_name, metric_val, epoch)
        #writer.add_scalar(name_with_mode, metric_val, epoch)
    
    def on_train_epoch_end(self, logs={}):
        for metric in self.track_epochwise:
            self.__log_on_tensorboard(
                metric_name=metric,
                logs=logs,
                epoch=logs['epoch_num'])
        return logs

    def on_val_epoch_end(self, logs={}):
        for metric in self.track_epochwise:
            self.__log_on_tensorboard(
                metric_name=metric,
                logs=logs,
                epoch=logs['epoch_num'],
                mode='val')
        return logs

    def on_train_batch_end(self, logs={}):
        batch_idx = logs['train']['batch_idx'] + logs['epoch_num']*logs['train_batches_per_epoch']
        for metric in self.track_batchwise:
            self.__log_on_tensorboard(
                metric_name=metric,
                logs=logs,
                epoch=batch_idx
            )
        return logs

    def on_val_batch_end(self, logs={}):
        batch_idx = logs['val']['batch_idx'] + logs['epoch_num']*logs['val_batches_per_epoch']
        for metric in self.track_batchwise:
            self.__log_on_tensorboard(
                metric_name=metric,
                logs=logs,
                epoch=batch_idx,
                mode='val')
        return logs
    
    def on_train_end(self, logs={}):
        for _, writer in self.writers.items():
            writer.flush()
            writer.close()
        return logs

class EarlyStopper(Callback):
    def __init__(self, model, metric_name, patience=3, decreasing=False, restore_best_weights=True):
        self.best_weights = None
        self.metric_name = metric_name
        self.model = model
        self.patience = patience
        self.decreasing = decreasing
        self.restore_best_weights = restore_best_weights
        self.best_value = -float('inf')
        self.current_patience = self.patience
    
    def __decrease_patience(self, metric_val):
        metric_val = -metric_val if self.decreasing else metric_val
        if metric_val > self.best_value:
            self.current_patience = self.patience
            self.best_value = metric_val
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(self.model.state_dict())
        else:
            self.current_patience -= 1
    
    def __early_stop(self, logs, mode='train'):
        if mode in self.metric_name:
            metric_val = logs[mode]['loss'] if 'loss' in self.metric_name else logs['metrics'][self.metric_name]
            self.__decrease_patience(metric_val)
            if self.current_patience == 0:
                logs['early_stop'] = True
                if self.restore_best_weights:
                    self.model.load_state_dict(self.best_weights)
        return logs
        
    def on_train_epoch_end(self, logs={}):
        logs = self.__early_stop(logs, mode='train')
        return logs

    def on_val_epoch_end(self, logs={}):
        logs = self.__early_stop(logs, mode='val')
        return logs

class CheckpointSaver(Callback):
    def __init__(self, model, optimizer, path, monitor='val_loss', decreasing=True):
        self.model = model
        self.optimizer = optimizer
        self.path = path
        self.metric_name = monitor
        self.decreasing = decreasing
        self.best_value = -float('inf')
        if not os.path.exists(path):
            os.mkdir(path)

    def __make_checkpoint(self, logs, mode='train'):
        if mode in self.metric_name:
            metric_val = logs[mode]['loss'] if 'loss' in self.metric_name else logs['metrics'][self.metric_name]
            metric_val = -metric_val if self.decreasing else metric_val
            if metric_val > self.best_value:
                self.best_value = metric_val
                torch.save({
                    'epoch': logs['epoch_num'],
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': logs['train']['loss'],
                    }, os.path.join(self.path, self.model.name + '.pt'))
        
    def on_train_epoch_end(self, logs={}):
        self.__make_checkpoint(logs, mode='train')
        return logs

    def on_val_epoch_end(self, logs={}):
        self.__make_checkpoint(logs, mode='val')
        return logs

class UnfreezeLayers(Callback):
    def __init__(self, layers, epoch_num):
        self.epoch_num = epoch_num
        self.layers = layers
    
    def on_train_epoch_end(self, logs={}):
        if logs['epoch_num'] + 1 == self.epoch_num:
            for layer in self.layers:
                for p in layer.parameters():
                    p.requires_grad = True
        return logs
