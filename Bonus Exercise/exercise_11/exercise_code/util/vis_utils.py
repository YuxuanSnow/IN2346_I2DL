"""Utils for visualizations in notebooks"""

import matplotlib.pyplot as plt


def show_all_keypoints(image, keypoints, pred_kpts=None):
    """Show image with predicted keypoints"""
    image = (image.clone() * 255).view(96, 96)
    plt.imshow(image, cmap='gray')
    keypoints = keypoints.clone() * 48 + 48
    plt.scatter(keypoints[:, 0], keypoints[:, 1], s=200, marker='.', c='m')
    if pred_kpts is not None:
        pred_kpts = pred_kpts.clone() * 48 + 48
        plt.scatter(pred_kpts[:, 0], pred_kpts[:, 1], s=200, marker='.', c='r')
    plt.show()
