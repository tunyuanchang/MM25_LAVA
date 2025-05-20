# visualization for comparison
# 3D skeleton ground truth and prediction
# cr: tunyuanchang

import numpy as np
import matplotlib.pyplot as plt

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
          'black', 'black', 'black', 'black',
          'orange', 'orange', 'orange',
          'green', 'green', 'green']

# Color settings
joint_color_target = 'blue'
joint_color_pred = 'red'
limb_color_target = 'black'
limb_color_pred = 'gray'

def draw_skeleton_comparison(pred_array, target_array, plt_show=True, save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)

    # Bounding box
    combined = np.vstack((pred_array, target_array))
    xdata, ydata, zdata = combined[:, 0], combined[:, 1], combined[:, 2]

    max_range = np.array([
        xdata.max() - xdata.min(),
        ydata.max() - ydata.min(),
        zdata.max() - zdata.min()
    ]).max()

    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (xdata.max() + xdata.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (ydata.max() + ydata.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (zdata.max() + zdata.min())

    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    # Plot target (blue joints, black limbs)
    ax.scatter(target_array[:, 0], target_array[:, 1], target_array[:, 2], c=joint_color_target, label='Target')
    for (start, end) in skeleton:
        ax.plot(
            [target_array[start, 0], target_array[end, 0]],
            [target_array[start, 1], target_array[end, 1]],
            [target_array[start, 2], target_array[end, 2]],
            color=limb_color_target, linestyle='-', linewidth=2
        )

    # Plot prediction (red joints, gray limbs)
    ax.scatter(pred_array[:, 0], pred_array[:, 1], pred_array[:, 2], c=joint_color_pred, label='Prediction')
    for (start, end) in skeleton:
        ax.plot(
            [pred_array[start, 0], pred_array[end, 0]],
            [pred_array[start, 1], pred_array[end, 1]],
            [pred_array[start, 2], pred_array[end, 2]],
            color=limb_color_pred, linestyle='--', linewidth=2
        )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_title('3D Skeleton Comparison')
    ax.legend()

    if plt_show:
        plt.show()

    if save_path is not None:
        fig.savefig(save_path)
        plt.close()
