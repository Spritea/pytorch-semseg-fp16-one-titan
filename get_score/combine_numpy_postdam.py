import numpy as np
from pathlib import Path
import natsort
from tqdm import tqdm
import cv2


def combine_one(np_list,np_path,height,width):
    np_one = np.load(np_list[0])
    channel, small_height, small_width = np_one.shape
    # np_one = np.reshape(np_one, [small_height, small_width, channel])
    np_large = np.zeros((height, width, channel), dtype=np.float32)
    row_res = height % small_height
    col_res = width % small_width
    img_row = int(height / small_height) if row_res == 0 else int(height / small_height) + 1
    img_col = int(width / small_width) if col_res == 0 else int(width / small_width) + 1

    for k in range(img_row):
        for j in range(img_col):
            p = np.load(np_list[j + k * img_col])
            # 把channel换到后面去,方便下面拼接时的维度设置
            p = np.reshape(p, [small_height, small_width, channel])
            if j + 1 == img_col and k + 1 < img_row and col_res > 0:
                p=p[:,(small_width-col_res):,:]
                # np_large[k * small_height:(k + 1) * small_height,
                # j * small_width:width, :] = p
            elif j + 1 < img_col and k + 1 == img_row and row_res > 0:
                p=p[(small_height-row_res):,:,:]
                # np_large[k * small_height:height,
                # j * small_width:(j + 1) * small_width, :] = p
            elif j + 1 == img_col and k + 1 == img_row and col_res > 0 and row_res > 0:
                p=p[(small_height-row_res):,(small_width-col_res):,:]
                # np_large[k * small_height:height,
                # j * small_width:width, :] = p
            np_large[k * small_height:(k + 1) * small_height,
            j * small_width:(j + 1) * small_width, :] = p
            #numpy自己会调节,使得(k + 1) * small_height不会超过最大范围

    out_path=out_path_prefix+'/'+np_path
    np.save(out_path,np_large)

#Postdam train18
id_list=['2_12','2_13','2_14','3_12','3_13','3_14','4_12','4_13','4_14','4_15',
         '5_12','5_13','5_14','5_15','6_12','6_13','6_14','6_15','7_12','7_13']

Numpy_Path = Path("/home/spl03/code/pytorch-semseg-fp16/test_out/potsdam/data14/numpy_softmax/mv3_1_true_2_res50")
Large_Path = Path("/home/spl03/code/pytorch-semseg-fp16/get_score/Postdam/val_gt_full")
Large_File = natsort.natsorted(list(Large_Path.glob("*.tif")), alg=natsort.PATH)
Large_Str = []
for j in Large_File:
    Large_Str.append(str(j))

for k in tqdm(range(len(id_list))):
    glob_target='*potsdam_'+id_list[k]+'_*.npy'
    Numpy_File = natsort.natsorted(list(Numpy_Path.glob(glob_target)), alg=natsort.PATH)
    Numpy_Str = []
    for i in Numpy_File:
        Numpy_Str.append(str(i))
    large_img=cv2.imread(Large_Str[k],1)
    height,width,_=large_img.shape

    out_path_prefix = "/home/spl03/code/pytorch-semseg-fp16/test_out/potsdam/data14/numpy_softmax/combined_numpy/mv3_1_true_2_res50"
    out_name='potsdam_'+id_list[k]+'_pred.npy'
    combine_one(Numpy_Str,out_name,width,height)




