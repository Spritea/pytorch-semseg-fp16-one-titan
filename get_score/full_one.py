import cv2 as cv
from get_score import util
import numpy as np
import time
from pathlib import Path
import natsort

from get_score.metrics_my import runningScore
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool


def load_image(path):
    image = cv.cvtColor(cv.imread(path, 1), cv.COLOR_BGR2RGB)
    return image

Tensor_Path = Path("/home/spl03/code/pytorch-semseg-fp16/test_out/vaihingen/data07_or_08/numpy_softmax/combined_numpy/mv3_1_true_2_res50")
Tensor_File = natsort.natsorted(list(Tensor_Path.glob("*.npy")), alg=natsort.PATH)
Tensor_Str = []
for j in Tensor_File:
    Tensor_Str.append(str(j))

GT_Path = Path("/home/spl03/code/pytorch-semseg-fp16/get_score/Vaihingen/val_gt_full")
GT_File = natsort.natsorted(list(GT_Path.glob("*.tif")), alg=natsort.PATH)
GT_Str = []
for j in GT_File:
    GT_Str.append(str(j))

th=0
# th=list(np.arange(0,0.99,0.02))
pre=[]
rec=[]
t = time.time()
running_metrics_val = runningScore(2)
# label_values_binary = [[0, 0, 0], [250, 250, 250]]
# label_values = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0],[0,0,0]]
# 注意不要[0,0,0],这是local evaluation.
label_values = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]


object_class=1
def compute_one(img_path,gt_path):
    gt = load_image(gt_path)
    # val_gt_erode paired with [0,0,0]label value
    # label order: R G B
    # num_classes = len(label_values)
    gt = util.reverse_one_hot(util.one_hot_it(gt, label_values))
    gt_binary=np.zeros(gt.shape,dtype=np.uint8)
    gt_binary[gt==object_class]=1
    output_image = img_path
    running_metrics_val.update(gt_binary, output_image)


def full_one():
    running_metrics_val.reset()
    for k in range(len(Tensor_Str)):
        lanes=np.load(Tensor_Str[k])
        lanes_one_channel = lanes[:,:,object_class]
        height, width= lanes_one_channel.shape
        pred = np.zeros((height, width), dtype=np.uint8)
        # pred[lanes_one_channel > th] = 4
        pred[lanes_one_channel > th] = 1
        # pred_3chan=np.repeat(pred.reshape(height, width, 1), 3, axis=2)
        compute_one(pred, GT_Str[k])

full_one()

# pool=ThreadPool(16)
# pool.map(full_one(),th)
# pool.close()
# pool.join()

acc, cls_pre, cls_rec, cls_f1, cls_iu, hist = running_metrics_val.get_scores()
tt = time.time() - t
pre.append(cls_pre[1])
rec.append(cls_rec[1])
pre_path='/home/spl03/code/pytorch-semseg-fp16/get_score/Vaihingen/PR/fcn8s/pre_'+str(object_class)+'.npy'
rec_path='/home/spl03/code/pytorch-semseg-fp16/get_score/Vaihingen/PR/fcn8s/rec_'+str(object_class)+'.npy'

# np.save(pre_path,pre)
# np.save(rec_path,rec)
print("cls pre")
print(cls_pre)
print("cls rec")
print(cls_rec)
print("time: %f" %tt)
