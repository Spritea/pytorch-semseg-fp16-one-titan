import cv2 as cv
from get_score import util
import numpy as np
import time
from pathlib import Path
import natsort

from get_score.metrics_my import runningScore
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import itertools

def load_image(path):
    image = cv.cvtColor(cv.imread(path, 1), cv.COLOR_BGR2RGB)
    return image

Tensor_Path = Path("/home/spl03/code/pytorch-semseg-fp16/test_out/vaihingen/data07_or_08/numpy_softmax/combined_numpy/deeplabv3_plus")
Tensor_File = natsort.natsorted(list(Tensor_Path.glob("*.npy")), alg=natsort.PATH)
Tensor_Str = []
for j in Tensor_File:
    Tensor_Str.append(str(j))

GT_Path = Path("/home/spl03/code/pytorch-semseg-fp16/get_score/Vaihingen/val_gt_full")
GT_File = natsort.natsorted(list(GT_Path.glob("*.tif")), alg=natsort.PATH)
GT_Str = []
for j in GT_File:
    GT_Str.append(str(j))
prefix='/home/spl03/code/pytorch-semseg-fp16/get_score/Vaihingen/PR/deeplabv3_plus'

# th=0.98
th=list(np.arange(0,0.99,0.02))
th.extend([0.99,0.999,0.9999,0.99999])
pre=[]
rec=[]
t = time.time()
running_metrics_val = runningScore(2)
# label_values = [[0, 0, 0], [250, 250, 250]]
# label_values = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0],[0,0,0]]
# 注意不要[0,0,0],这是local evaluation.
label_values = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]

object_class=[0,1,2,3,4]
def compute_one(img_path,gt_path,object):
    gt = load_image(gt_path)
    # val_gt_erode paired with [0,0,0]label value
    # label order: R G B
    # num_classes = len(label_values)
    gt = util.reverse_one_hot(util.one_hot_it(gt, label_values))
    gt_binary = np.zeros(gt.shape, dtype=np.uint8)
    gt_binary[gt == object] = 1
    output_image = img_path
    running_metrics_val.update(gt_binary, output_image)


def backbone(k,th,object):
    lanes = np.load(Tensor_Str[k])
    lanes_one_channel = lanes[:, :, object]
    height, width = lanes_one_channel.shape
    pred = np.zeros((height, width), dtype=np.uint8)
    # pred[lanes_one_channel > th] = 4
    pred[lanes_one_channel > th] = 1
    # pred_3chan=np.repeat(pred.reshape(height, width, 1), 3, axis=2)
    compute_one(pred, GT_Str[k],object)

def full_one(th,object):
    running_metrics_val.reset()

    pool=ThreadPool(26)
    pool.starmap(backbone,zip(range(len(Tensor_Str)),itertools.repeat(th),itertools.repeat(object)))
    pool.close()
    pool.join()

    acc, cls_pre, cls_rec, cls_f1, cls_iu, hist = running_metrics_val.get_scores()
    pre.append(cls_pre[1])
    rec.append(cls_rec[1])
    print("cls pre")
    print(cls_pre)
    print("cls rec")
    print(cls_rec)

for object in object_class:
    for item in th:
        print("class: %d"%object)
        print("th: %f"%th)
        full_one(item,object)

    pre_path = prefix + '/' + 'pre_' + str(object) + '.npy'
    rec_path = prefix + '/' + 'rec_' + str(object) + '.npy'

    np.save(pre_path, pre)
    np.save(rec_path, rec)

tt = time.time() - t

print("time: %f" %tt)
