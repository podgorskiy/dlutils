from yacs.config import CfgNode as CN


__all__ = ['get_default_cfg']


_C = CN()

_C.OUTPUT_DIR = "results"

_C.DATASET = CN()
_C.DATASET.PATH = ''

_C.MODEL = CN()

_C.TRAIN = CN()

_C.TRAIN.BASE_LEARNING_RATE = 0.05
_C.TRAIN.LEARNING_DECAY_RATE = 0.1
_C.TRAIN.LEARNING_DECAY_STEPS = [1000, 2000]
_C.TRAIN.TRAIN_EPOCHS = 3000


_C.TRAIN.SNAPSHOT_FREQ = 100

_C.TRAIN.REPORT_FREQ = 30


def get_default_cfg():
    return _C.clone()
