import sys, os
import torch
import argparse
import timeit
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.utils import convert_state_dict

import yaml
from pathlib import Path
import natsort
import cv2 as cv

def test(args,cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[: model_file_name.find("_")]

    IMG_Path=Path(args.img_path)
    IMG_File=natsort.natsorted(list(IMG_Path.glob("*.png")),alg=natsort.PATH)
    IMG_Str=[]
    for i in IMG_File:
        IMG_Str.append(str(i))
    # Setup image
    print("Read Input Image from : {}".format(args.img_path))

    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset,config_file=cfg)
    loader = data_loader(data_path, is_transform=True, img_norm=args.img_norm)
    n_classes = loader.n_classes

    # Setup Model
    model = get_model(cfg['model'], n_classes)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    # state=torch.load(args.model_path)["model_state"]
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    for j in tqdm(range(len(IMG_Str))):
        img_path=IMG_Str[j]
        img = misc.imread(img_path)
        # img = img[:, :, ::-1]
        img = img.astype(np.float64)
        # img -= loader.mean
        if args.img_norm:
            img = img.astype(float) / 255.0

        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img).float()
        images = img.to(device)
        outputs = model(images)
        out_path_pt = "test_out/vaihingen/data07_or_08/mv3_1_true_2_res50/pt/" + Path(img_path).stem + ".pt"
        torch.save(outputs,out_path_pt)
        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)

        decoded = loader.decode_segmap(pred)
        out_path="test_out/vaihingen/data07_or_08/mv3_1_true_2_res50/pic/"+Path(img_path).name

        decoded_bgr=cv.cvtColor(decoded,cv.COLOR_RGB2BGR)
        # misc.imsave(out_path, decoded)
        cv.imwrite(out_path,decoded_bgr)

    # print("Classes found: ", np.unique(pred))
    # print("Segmentation Mask Saved at: {}".format(args.out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="pretrain/data07_or_08/mv3_res50_my_best_model.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/mv3_1_true_2_res50.yml",
        help="Configuration file to use"
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        default="my",
        help="Dataset to use ['pascal, camvid, ade20k etc']",
    )

    parser.add_argument(
        "--img_norm",
        dest="img_norm",
        action="store_true",
        help="Enable input image scales normalization [0, 1] \
                              | True by default",
    )
    parser.add_argument(
        "--no-img_norm",
        dest="img_norm",
        action="store_false",
        help="Disable input image scales normalization [0, 1] |\
                              True by default",
    )
    parser.set_defaults(img_norm=True)

    parser.add_argument(
        "--dcrf",
        dest="dcrf",
        action="store_true",
        help="Enable DenseCRF based post-processing | \
                              False by default",
    )
    parser.add_argument(
        "--no-dcrf",
        dest="dcrf",
        action="store_false",
        help="Disable DenseCRF based post-processing | \
                              False by default",
    )
    parser.set_defaults(dcrf=False)

    parser.add_argument(
        "--img_path", nargs="?", type=str,
        default="dataset/08-Vahingen-train23-rotate-precode-07edit/val", help="Path of the input image"
    )
    parser.add_argument(
        "--out_path",
        nargs="?",
        type=str,
        default="tk.png",
        help="Path of the output segmap",
    )

    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp)
    test(args,cfg)
