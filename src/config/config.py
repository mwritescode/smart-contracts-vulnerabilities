from yacs.config import CfgNode as ConfigurationNode

# YACS overwrite these settings using YAML
__C = ConfigurationNode()

__C.MODEL = ConfigurationNode()
__C.MODEL.NAME = 'resnet' # ADD MODEl CONFIGURATIONS AS THE CODING PROGRESSES
__C.MODEL.N_CLASSES = 5

__C.DATASET = ConfigurationNode()
__C.DATASET.RGB_IMAGES = True
__C.DATASET.IMG_SHAPE = 224
__C.DATASET.USE_IMAGENET_STATS = False
__C.DATASET.AUGUMENTATION = False
__C.DATASET.BINARY_LABELS = False
__C.DATASET.MAX_SEQ_LEN = 512

__C.DATASET.LOADER = ConfigurationNode()
__C.DATASET.LOADER.BATCH_SIZE = 16

__C.TRAINING = ConfigurationNode()
__C.TRAINING.N_EPOCHS = 30
__C.TRAINING.TRAIN_FROM_SCRATCH = False
__C.TRAINING.LAYERS_TO_FINETUNE = 6
__C.TRAINING.LOSS = 'binary_crossentropy'

__C.TRAINING.OPTIMIZER = ConfigurationNode()
__C.TRAINING.OPTIMIZER.NAME = 'adam'
__C.TRAINING.OPTIMIZER.LR = 1e-3
__C.TRAINING.OPTIMIZER.WEIGHT_DECAY = 0.0001
__C.TRAINING.OPTIMIZER.MOMENTUM = 0.9

__C.TRAINING.EARLY_STOPPING = ConfigurationNode()
__C.TRAINING.EARLY_STOPPING.USE = True
__C.TRAINING.EARLY_STOPPING.MONITOR = 'val_loss'
__C.TRAINING.EARLY_STOPPING.DECREASING = True
__C.TRAINING.EARLY_STOPPING.PATIENCE = 10

__C.TRAINING.CHECKPOINTS = ConfigurationNode()
__C.TRAINING.CHECKPOINTS.USE = True
__C.TRAINING.CHECKPOINTS.MONITOR = 'val_acc'
__C.TRAINING.CHECKPOINTS.DECREASING = False
__C.TRAINING.CHECKPOINTS.PATH = 'checkpoints/<config_name>.pkl'

__C.TRAINING.LOGGER = ConfigurationNode()
__C.TRAINING.LOGGER.USE = True
__C.TRAINING.LOGGER.RUN_TAG = '<config_name>'

__C.TRAINING.TRACK_METRICS = ConfigurationNode()
__C.TRAINING.TRACK_METRICS.USE = True
__C.TRAINING.TRACK_METRICS.NAMES = ('f1', 'precision', 'recall')
__C.TRAINING.TRACK_METRICS.AVERAGE = ['macro', 'micro'] #Optionally change to/add micro and weighted

def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    return __C.clone()

def save_cfg_default():
    """Save in a YAML file the default version of the configuration file, in order to provide a template to be modified."""
    with open('src/config/experiments/default.yaml', 'w') as f:
        f.write(__C.dump())
        f.flush()
        f.close()
