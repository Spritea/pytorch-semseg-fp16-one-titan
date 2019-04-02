import numpy as np
import torch
import torch.nn.functional as F

t1=torch.load("top_mosaic_09cm_area16_0000.pt")
t2=F.softmax(t1)
t3=t2.data.cpu().numpy()
# a=t1.data.max(1)
# b=a[1]
# c=t1[0,:,0,0]
t4=t3[:,0,:,:]
t5=np.squeeze(t4)
a=[0<t5]
b=[t5<0.5]
c=a and b
t5[t5>=0.5]=0
t5[0<t5 and t5<0.5]=1
print("kk")


def decode_segmap(self, temp, plot=False):
    Imps = [255, 255, 255]
    Building = [0, 0, 255]
    Lowvg = [0, 255, 255]
    Tree = [0, 255, 0]
    Car = [255, 255, 0]
    bg = [255, 0, 0]

    label_colours = np.array(
        [
            Imps,
            Building,
            Lowvg,
            Tree,
            Car,
            bg,
        ]
    )
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, self.n_classes):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    # rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb = np.zeros((temp.shape[0], temp.shape[1], 3), dtype=np.uint8)
    # rgb[:, :, 0] = r / 255.0
    # rgb[:, :, 1] = g / 255.0
    # rgb[:, :, 2] = b / 255.0
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb