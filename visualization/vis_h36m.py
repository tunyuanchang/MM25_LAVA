# visualization for Human3.6m
# 3D skeleton ground truth
# cr: tunyuanchang

import json
import numpy as np
import matplotlib.pyplot as plt

subject_id = 1
action_id = 2
subaction_id = 1
frame_id = 10

# joint set
joint_num = 17

joints_name = ('pelvis', 'R_hip', 'R_knee', 'R_ankle', 
               'L_hip', 'L_knee', 'L_ankle', 
               'spine', 'thorax', 'head', 'head_top', 
               'L_Shoulder', 'L_Elbow', 'L_Wrist',
               'R_Shoulder', 'R_Elbow', 'R_Wrist')


skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), # axis
            (0, 1), (1, 2), (2, 3), # right leg
            (0, 4), (4, 5), (5, 6), # left leg
            (8, 14), (14, 15), (15, 16), # right arm
            (8, 11), (11, 12), (12, 13)) # left arm

colors = ['black',
          'blue', 'blue', 'blue',
          'red', 'red', 'red',
          'purple', 'purple', 'purple', 'purple',
          'orange', 'orange', 'orange',
          'green', 'green', 'green']

def draw_skeleton(array, plt_show=True, save_path='vis.eps'):
    fig = plt.figure(figsize=(6.4, 4.8), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    # fig.subplots_adjust(left=0.05, right=4.75,top=6.35,bottom=0)
    ax.grid(False)

    ### Plot fakebbox for margin
    xdata = array[:, 0]
    ydata = array[:, 1]
    zdata = array[:, 2]
    
    max_range= np.array([xdata.max() - xdata.min(), ydata.max() - ydata.min(), zdata.max() - zdata.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (xdata.max() + xdata.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (ydata.max() + ydata.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (zdata.max() + zdata.min())

    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')
    ###
    
    # Plot joints
    ax.scatter(array[0, 0], array[0, 1], array[0, 2], c=colors[0], marker='+', s=80, label='Pelvis')
    ax.scatter(array[1:4, 0], array[1:4, 1], array[1:4, 2], c=colors[1:4], label='Right Leg')
    ax.scatter(array[4:7, 0], array[4:7, 1], array[4:7, 2], c=colors[4:7], label='Left Leg')
    ax.scatter(array[7:11, 0], array[7:11, 1], array[7:11, 2], c=colors[7:11], label='Central Joint')
    ax.scatter(array[11:14, 0], array[11:14, 1], array[11:14, 2], c=colors[11:14], label='Left Arm')
    ax.scatter(array[14:, 0], array[14:, 1], array[14:, 2], c=colors[14:], label='Right Arm')
    
    # Plot skeleton
    for (start, end) in skeleton:
        ax.plot([array[start, 0], array[end, 0]], [array[start, 1], array[end, 1]], [array[start, 2], array[end, 2]], color='gray')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Show plot
    # if plt_show:
    #     plt.show()
    
    # plt.tight_layout()
    # plt.show()
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(0, 0.75),  # slightly lower than top
        bbox_transform=ax.transAxes
    )

    if save_path is not None:
        fig.savefig(save_path)
        plt.close()


def print_item(subject_id, act_id, subact_id, frame_id):
    # dataset: https://github.com/mks0601/3DMPPE_POSENET_RELEASE
    # https://drive.google.com/drive/folders/1r0B9I3XxIIW_jsXjYinDpL6NFcxTZart?usp=sharing
    json_path = f"../Human36M/joints/Human36M_subject{subject_id}_joint_3d.json"

    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    array = np.array(data[str(act_id)][str(subact_id)][str(frame_id)]) # [17, 3]
    
    # Pelvis-centric
    array = array - array[0]

    draw_skeleton(array)

    return

if __name__ == '__main__':
    print_item(subject_id, action_id, subaction_id, frame_id)