# visualization for H3WB
# RGB images to 3D skeleton
# cr: tunyuanchang

import torch
import json
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "/media/tunyuan/Backup/H3WB/json/RGBto3D_train.json"

def data_loader(data_path):
    id_list = []
    input_list = []
    target_list = []
    bbox_list = []
    
    data = json.load(open(data_path))
    print(len(data))
    length = 1 # len(data)
    for i in range(length):
        id_list.append(i)
        sample_3d = torch.zeros(1, 133, 3)
        bbox = torch.zeros(1,4)
        bbox[0, 0] = int(data[str(i)]['bbox']['x_min'])
        bbox[0, 1] = int(data[str(i)]['bbox']['y_min'])
        bbox[0, 2] = int(data[str(i)]['bbox']['x_max'])
        bbox[0, 3] = int(data[str(i)]['bbox']['y_max'])
        bbox_list.append(bbox)
        for j in range(133):
            sample_3d[0, j, 0] = data[str(i)]['keypoints_3d'][str(j)]['x']
            sample_3d[0, j, 1] = data[str(i)]['keypoints_3d'][str(j)]['y']
            sample_3d[0, j, 2] = data[str(i)]['keypoints_3d'][str(j)]['z']
        input_list.append(data[str(i)]['image_path'])
        target_list.append(sample_3d)
    return id_list, input_list, target_list, bbox_list

def get_limb(X, Y, Z=None, id1=0, id2=1):
    return np.concatenate((np.expand_dims(X[id1], 0), np.expand_dims(X[id2], 0)), 0), \
           np.concatenate((np.expand_dims(Y[id1], 0), np.expand_dims(Y[id2], 0)), 0), \
           np.concatenate((np.expand_dims(Z[id1], 0), np.expand_dims(Z[id2], 0)), 0)

# draw wholebody skeleton
# conf: which joint to draw, conf=None draw all
def draw_skeleton(X, conf=None, plt_show=True, save_path=None, inverse_z=True):
    # pelvis-centric
    X = (X - (X[:,11,:]+X[:, 12,:])/2.0).numpy()
    list_branch_head = [(0,1),(1,3),(0,2),(2,4), (59,64), (65,70),(71, 82),
                        (71,83),(77,87),(77,88),(88,89),(89,90),(71,90)]
    for i in range(16):
        list_branch_head.append((23+i, 24+i))
    for i in range(4):
        list_branch_head.append((40+i, 41+i))
        list_branch_head.append((45+i, 46+i))
        list_branch_head.append((54+i, 55+i))
        list_branch_head.append((83+i, 84+i))
    for i in range(3):
        list_branch_head.append((50+i, 51+i))
    for i in range(5):
        list_branch_head.append((59+i, 60+i))
        list_branch_head.append((65+i, 66+i))
    for i in range(11):
        list_branch_head.append((71+i, 72+i))

    list_branch_left_arm = [(5,7),(7,9),(9,91),(91,92),(93,96),(96,100),(100,104),(104,108),(91,108)]
    for i in range(3):
        list_branch_left_arm.append((92+i,93+i))
        list_branch_left_arm.append((96+i,97+i))
        list_branch_left_arm.append((100+i,101+i))
        list_branch_left_arm.append((104+i,105+i))
        list_branch_left_arm.append((108+i,109+i))
        
    list_branch_right_arm = [(6,8),(8,10),(10,112),(112,113),(114,117),(117,121),(121,125),(125,129),(112,129)]
    for i in range(3):
        list_branch_right_arm.append((113+i, 114+i))
        list_branch_right_arm.append((117+i, 118+i))
        list_branch_right_arm.append((121+i, 122+i))
        list_branch_right_arm.append((125+i, 126+i))
        list_branch_right_arm.append((129+i, 130+i))
        
    list_branch_body = [(5,6),(6,12),(11,12),(5,11)]
    list_branch_right_foot = [(12,14),(14,16),(16,20),(16,21),(16,22)]
    list_branch_left_foot = [(11,13),(13,15),(15,17),(15,18),(15,19)]

    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.elev = 10
    ax.grid(False)

    if inverse_z:
        zdata = -X[0, :, 1]
    else:
        zdata = X[0, :, 1]
    xdata = X[0, :, 0]
    ydata = X[0, :, 2]
    
    if conf is not None:
        xdata*=conf[0,:].numpy()
        ydata*=conf[0,:].numpy()
        zdata*=conf[0,:].numpy()

    # plot fakebbox for margin
    max_range= np.array([xdata.max() - xdata.min(), ydata.max() - ydata.min(), zdata.max() - zdata.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (xdata.max() + xdata.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (ydata.max() + ydata.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (zdata.max() + zdata.min())

    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    # plot keypoints
    ax.scatter(xdata, ydata, zdata, c='r')

    # plot limb
    for (id1, id2) in list_branch_head:
        if ((conf is None) or ((conf[0,id1]>0.0) and (conf[0,id2]>0.0))):
            x, y, z = get_limb(xdata, ydata, zdata, id1, id2)
            ax.plot(x, y, z, color='red')
    for (id1, id2) in list_branch_body:
        if ((conf is None) or ((conf[0, id1] > 0.0) and (conf[0, id2] > 0.0))):
            x, y, z = get_limb(xdata, ydata, zdata, id1, id2)
            ax.plot(x, y, z, color='orange')
    for (id1, id2) in list_branch_left_arm:
        if ((conf is None) or ((conf[0,id1]>0.0) and (conf[0,id2]>0.0))):
            x, y, z = get_limb(xdata, ydata, zdata, id1, id2)
            ax.plot(x, y, z, color='blue')
    for (id1, id2) in list_branch_right_arm:
        if ((conf is None) or ((conf[0,id1]>0.0) and (conf[0,id2]>0.0))):
            x, y, z = get_limb(xdata, ydata, zdata, id1, id2)
            ax.plot(x, y, z, color='violet')
    for (id1, id2) in list_branch_left_foot:
        if ((conf is None) or ((conf[0,id1]>0.0) and (conf[0,id2]>0.0))):
            x, y, z = get_limb(xdata, ydata, zdata, id1, id2)
            ax.plot(x, y, z, color='cyan')
    for (id1, id2) in list_branch_right_foot:
        if ((conf is None) or ((conf[0,id1]>0.0) and (conf[0,id2]>0.0))):
            x, y, z = get_limb(xdata, ydata, zdata, id1, id2)
            ax.plot(x, y, z, color='pink')

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    if plt_show:
        plt.show()
    if save_path is not None:
        fig.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    # keypoint ref: https://github.com/wholebody3d/wholebody3d/blob/main/imgs/Fig2_anno.png
    # visualization ref: https://github.com/wholebody3d/wholebody3d/blob/main/utils/utils.py
    id_list, input_list, target_list, bbox_list = data_loader(DATA_PATH)
    body_index = list(range(0, 17))
    foot_index = list(range(17, 23))
    face_index = list(range(23, 91))
    hands_index = list(range(91, 133))

    visual_index = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    conf = torch.zeros((1, 133), dtype=float)
    
    for i in visual_index: conf[0,i] = 1.0
    print(input_list[0])
    draw_skeleton(target_list[0], conf = conf, plt_show=True)