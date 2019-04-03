import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import natsort
from tqdm import tqdm

Tensor_Path = Path("/home/spl03/code/pytorch-semseg-fp16/test_out/potsdam/data14/mv3_1_true_2_res50/pt")
Tensor_File = natsort.natsorted(list(Tensor_Path.glob("*.pt")), alg=natsort.PATH)
Tensor_Str = []
for j in Tensor_File:
    Tensor_Str.append(str(j))

prefix="/home/spl03/code/pytorch-semseg-fp16/test_out/potsdam/data14/numpy_softmax/mv3_1_true_2_res50/"
for k in tqdm(range(len(Tensor_Str))):
    tensor=torch.load(Tensor_Str[k])
    tensor=F.softmax(tensor)
    numpy_array=np.squeeze(tensor.data.cpu().numpy(), axis=0)
    numpy_name=prefix+Path(Tensor_Str[k]).stem + ".npy"
    np.save(numpy_name,numpy_array)
