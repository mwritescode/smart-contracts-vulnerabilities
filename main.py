import torch
import numpy as np

from torch import nn
from torch import optim
from torchinfo import summary
from datasets import load_dataset
from torchvision import transforms
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.nn.functional import normalize
from sklearn.metrics import accuracy_score

from src.config import config
from src.data.stats import GetMeanStd
from src.utils.registry import REGISTRY
from src.utils.trainer import Trainer
from src.modeling.solver.loss import SigmoidFocalLoss
from src.modeling.network import MultitaskModel, Head
from src.modeling.network.backbone import ResNet1D, ResNetModel, InceptionModel
from src.data.transform import generate_image_and_binary_label, generate_image_and_label
from src.data.transform import generate_signal_and_binary_label, generate_signal_and_label
from src.utils.callbacks import MetricsCallback, EarlyStopper, TensorBoardLogger, CheckpointSaver

IMAGENET_MEAN = [0.485, 0.456, 0.406] 
IMAGENET_STD = [0.229, 0.224, 0.225]

def train_pipeline(args):
    train_ds = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='train', ignore_verifications=True)
    val_ds = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='validation', ignore_verifications=True)


    train_ds = train_ds.filter(lambda elem: elem['bytecode'] != '0x')
    val_ds = val_ds.filter(lambda elem: elem['bytecode'] != '0x')

    CFG_PATH = args.cfg_path

    cfg = config.get_cfg_defaults()
    cfg.merge_from_file(CFG_PATH)
    cfg.freeze()
    
    if cfg.DATASET.RGB_IMAGES and cfg.DATASET.BINARY_LABELS:
        map_func = generate_image_and_binary_label
    elif cfg.DATASET.RGB_IMAGES and not cfg.DATASET.BINARY_LABELS:
        map_func = generate_image_and_label
    elif not cfg.DATASET.RGB_IMAGES and cfg.DATASET.BINARY_LABELS:
        map_func = generate_signal_and_binary_label
    else:
        map_func = generate_signal_and_label

    train_ds = train_ds.map(map_func, remove_columns=['address', 'source_code', 'bytecode', 'slither'])
    val_ds = val_ds.map(map_func, remove_columns=['address', 'source_code', 'bytecode', 'slither'])

    max_len = cfg.DATASET.MAX_SEQ_LEN

    if cfg.DATASET.RGB_IMAGES:
        img_size = cfg.DATASET.IMG_SHAPE

        if cfg.DATASET.USE_IMAGENET_STATS:
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        else:
            get_stats = GetMeanStd(train_ds, batch_size=16, img_size=img_size)
            mean, std = get_stats()

        img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def img_label_to_tensor(examples):
        if cfg.DATASET.RGB_IMAGES:
            examples['image'] = [img_transform(elem) for elem in examples['image']]
        else:
            examples['image'] = [np.pad(img, pad_width=(0, max_len - len(img))) if len(img) < max_len else img[:max_len] for img in examples['image']]
            examples['image'] = [torch.unsqueeze(normalize(torch.tensor(img).float(), dim=0), dim=0) for img in examples['image']]
        
        if cfg.DATASET.BINARY_LABELS:
            examples['label'] = torch.unsqueeze(examples['label'], -1)
        else:
            examples['label'] = torch.tensor(examples['label'])
        return examples

    train_ds.set_transform(img_label_to_tensor)
    val_ds.set_transform(img_label_to_tensor)

    num_cls = cfg.MODEL.N_CLASSES
    model_name = cfg.MODEL.NAME

    if 'multitask' in model_name:
        backbone_name = model_name.split('_')[1:]
        backbone = REGISTRY[backbone_name](classify=False, num_classes=num_cls)
        model = MultitaskModel(backbone=backbone, head=Head)
        train_heper = REGISTRY['multitask_train_helper']
    else:
        model = REGISTRY[model_name](num_classes=num_cls)
        train_heper = REGISTRY['inception_train_helper'] if 'inception' in model_name else REGISTRY['default_train_helper']
    
    if not cfg.TRAINING.TRAIN_FROM_SCRATCH:
        param_groups = model.get_layer_groups()
        for param in param_groups['feature_extractor'][:-cfg.TRAINING.LAYERS_TO_FINETUNE]:
            param.requires_grad = False

    print(summary(model))

    batch_size = cfg.DATASET.LOADER.BATCH_SIZE

    loader_train = DataLoader(train_ds,
                        batch_size=batch_size,
                        drop_last=True,
                        shuffle=True)
    loader_val = DataLoader(val_ds,
                        batch_size=batch_size,
                        drop_last=True,
                        shuffle=False)

    trainer = Trainer(model=model, train_dataloader=loader_train, val_dataloader=loader_val, train_helper=train_heper)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=cfg.TRAINING.OPTIMIZER.LR,
        weight_decay=cfg.TRAINING.OPTIMIZER.WEIGHT_DECAY)

    if 'crossentropy' not in cfg.TRAINING.LOSS:
        criterion = REGISTRY[cfg.TRAINING.LOSS]()
    else:
        criterion = nn.BCEWithLogitsLoss()

    trainer.compile(loss=criterion, optimizer=optimizer, metrics={'acc': accuracy_score})

    callbacks = []

    if cfg.TRAINING.TRACK_METRICS.USE:
        metrics = {}
        for avg in cfg.TRAINING.TRACK_METRICS.AVERAGE:
            print(avg)
            metrics.update({avg + '_' + metric: REGISTRY[metric](average=avg, labels=np.arange(0, num_cls)) for metric in cfg.TRAINING.TRACK_METRICS.NAMES})
        callbacks.append(MetricsCallback(metrics=metrics))

    if cfg.TRAINING.LOGGER.USE:
        add_to_logging = [] if not cfg.TRAINING.TRACK_METRICS.USE else metrics.keys()
        callbacks.append(TensorBoardLogger(
            track_epochwise=['loss', 'acc', *add_to_logging], 
            run_tag=cfg.TRAINING.LOGGER.RUN_TAG))

    if cfg.TRAINING.EARLY_STOPPING.USE:
        callbacks.append(EarlyStopper(
            model=model, 
            metric_name=cfg.TRAINING.EARLY_STOPPING.MONITOR, 
            decreasing=cfg.TRAINING.EARLY_STOPPING.DECREASING, 
            restore_best_weights=True, 
            patience=cfg.TRAINING.EARLY_STOPPING.PATIENCE))


    if cfg.TRAINING.CHECKPOINTS.USE:
        callbacks.append(CheckpointSaver(
            model=model, 
            optimizer=optimizer, 
            monitor=cfg.TRAINING.CHECKPOINTS.MONITOR, 
            decreasing=cfg.TRAINING.CHECKPOINTS.DECREASING, 
            path=cfg.TRAINING.CHECKPOINTS.PATH))

    trainer.fit(epochs=cfg.TRAINING.N_EPOCHS, callbacks=callbacks)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('cfg_path', help='Path of the model\'s configuration file')
    args = args.parse_args()
    train_pipeline(args)