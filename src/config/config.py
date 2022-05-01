from yacs.config import CfgNode as ConfigurationNode

# YACS overwrite these settings using YAML
__C = ConfigurationNode()

__C.INFURA_URI = 'https://mainnet.infura.io/v3/XXXXXXXXXXXXXXX'
__C.ETHERSCAN_API = 'XXXXXXXXXXXXXXX'

__C.MODEL = ConfigurationNode()
__C.MODEL.NAME = 'resnet18' # ADD MODEl CONFIGURATIONS AS THE CODING PROGRESSES
__C.MODEL.N_CLASSES = 6

__C.DATASET = ConfigurationNode()
__C.DATASET.NAME = 'smartbugs_wild'
__C.DATASET.RESIZE_SHAPE = (224, 224)
__C.DATASET.USE_IMAGENET_STATS = False
__C.DATASET.RGB_IMAGES = True
__C.DATASET.AUGUMENTATION = False

__C.DATASET.LOADER = ConfigurationNode()
__C.DATASET.LOADER.BATCH_SIZE = 16

__C.TRAINING = ConfigurationNode()
__C.TRAINING.N_EPOCHS = 30
__C.TRAINING.UNFREEZE_LAYERS = True
__C.TRAINING.UNFREEZE_LAYERS_AT_EPOCH = 5

__C.TRAINING.OPTIMIZER = ConfigurationNode()
__C.TRAINING.OPTIMIZER.NAME = 'adam'
__C.TRAINING.OPTIMIZER.LRS = (1e-3, 1e-2)
__C.OPTIMIZER.WEIGHT_DECAY = 0.001

__C.TRAINING.EARLY_STOPPING = ConfigurationNode()
__C.TRAINING.EARLY_STOPPING.USE = True
__C.TRAINING.EARLY_STOPPING.MONITOR = 'val_acc'
__C.TRAINING.EARLY_STOPPING.DECREASING = False
__C.TRAINING.EARLY_STOPPING.PATIENCE = 3

__C.TRAINING.CHECKPOINTS = ConfigurationNode()
__C.TRAINING.CHECKPOINTS.USE = True
__C.TRAINING.CHECKPOINTS.MONITOR = 'val_acc'
__C.TRAINING.CHECKPOINTS.DECREASING = False
__C.TRAINING.CHECKPOINTS.PATH = 'checkpoints'
__C.TRAINING.CHECKPOINTS.FILENAME = __C.MODEL.NAME

__C.TRAINING.LOGGER = ConfigurationNode()
__C.TRAINING.LOGGER.USE = True
__C.TRAINING.LOGGER.RUN_TAG = ''

__C.TRAINING.TRACK_METRICS = ConfigurationNode()
__C.TRAINING.TRACK_METRICS.USE = True
__C.TRAINING.TRACK_METRICS.NAMES = ('f1', 'precision', 'recall')
__C.TRAINING.TRACK_METRICS.AVERAGE = ('macro') #Optionally change to/add micro

def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    return __C.clone()