# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
# import net.resnet_atrous as atrousnet
from ptsemseg.models.xception import xception

def build_backbone(backbone_name, pretrained=True, os=16):
	if backbone_name == 'xception' or backbone_name == 'Xception':
		net = xception(pretrained=pretrained, os=os)
		return net
	else:
		raise ValueError('backbone.py: The backbone named %s is not supported yet.'%backbone_name)
	

	

