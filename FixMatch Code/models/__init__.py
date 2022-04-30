from .wideresnet import *
from .wideresnet_lk import *
from .resnext import *

WRN_MODELS = {
        'WideResNet':WideResNet,
        'WideResNet_Lk': WideResNet_Lk,
        'CifarResNeXt':CifarResNeXt
        }
