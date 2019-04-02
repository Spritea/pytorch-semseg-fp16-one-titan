import copy
import torchvision.models as models

from ptsemseg.models.fcn import *
from ptsemseg.models.segnet import *
from ptsemseg.models.unet import *
from ptsemseg.models.pspnet import *
from ptsemseg.models.icnet import *
from ptsemseg.models.linknet import *
from ptsemseg.models.frrn import *

from ptsemseg.models.Refine import rf50
from ptsemseg.models.MV1_7 import MV1_7_ResNet50
from ptsemseg.models.MV1_5_1 import MV1_5_1_ResNet50
from ptsemseg.models.gcn import GCN
from ptsemseg.models.MV3_1_1 import MV3_1_1_ResNet50
from ptsemseg.models.MV3_1_true_2 import MV3_1_true_2_ResNet50

from ptsemseg.models.deeplabv3plus import deeplabv3plus
from ptsemseg.models.deeplabv3_os16_MG import DeepLabV3_MG
from ptsemseg.models.deeplabv3_os16_MG_plus import DeepLabV3_MG_plus


def get_model(model_dict, n_classes):
    name = model_dict['arch']
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop('arch')

    if name in ["frrnA", "frrnB"]:
        model = model(n_classes, **param_dict)

    elif name in ["fcn32s", "fcn16s", "fcn8s"]:
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "segnet":
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "unet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "pspnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "icnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "icnetBN":
        model = model(n_classes=n_classes, **param_dict)
    elif name=="refinenet50":
        model=model(num_classes=n_classes,imagenet=True,pretrained=False,**param_dict)
    elif name=="mv3_res50":
        model=model(num_classes=n_classes, **param_dict)
    elif name == "mv1_res50":
        model = model(num_classes=n_classes, **param_dict)
    elif name=="gcn":
        model=model(num_classes=n_classes, **param_dict)

    elif name == "deeplabv3p_os16":
        model = model(num_classes=n_classes, **param_dict)
    elif name=="deeplabv3_os16_MG":
        model=model(num_classes=n_classes, **param_dict)
    elif name=="deeplabv3_os16_MG_plus":
        model=model(num_classes=n_classes, **param_dict)

    else:
        model = model(n_classes=n_classes, **param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "fcn32s": fcn32s,
            "fcn8s": fcn8s,
            "fcn16s": fcn16s,
            "unet": unet,
            "segnet": segnet,
            "pspnet": pspnet,
            "icnet": icnet,
            "icnetBN": icnet,
            "linknet": linknet,
            "frrnA": frrn,
            "frrnB": frrn,
            "refinenet50": rf50,
            # "deeplabv3":Res_Deeplab,
            "deeplabv3p_os16": deeplabv3plus,
            "mv1_res50": MV3_1_1_ResNet50,
            #"mv3_res50":MV3_1_1_ResNet50,
            "mv3_res50": MV3_1_true_2_ResNet50,
            "gcn":GCN,
            "deeplabv3_os16_MG": DeepLabV3_MG,
            "deeplabv3_os16_MG_plus": DeepLabV3_MG_plus,
        }[name]
    except:
        raise("Model {} not available".format(name))
