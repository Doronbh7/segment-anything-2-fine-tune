
import numpy as np
from matplotlib import pyplot as plt
import torch
from copy import deepcopy
from typing import Tuple




class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(0, 1))
    union = torch.sum(pred_mask, dim=(0, 1)) + torch.sum(gt_mask, dim=(0, 1)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    return batch_iou

#return the mask with the best score .Return mask,score
def best_score_mask(masks,scores):
    num_columns = masks.shape[0]
    num=0
    for col_idx in range(num_columns):
        mask=masks[col_idx,:,:]
        score=scores[col_idx]

        if col_idx==0:
            best_score = score
            best_mask = mask
            num = col_idx

        else:
            if score > best_score:
                best_score = score
                best_mask = mask
                num=col_idx
    return best_mask,best_score,num


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)






def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size

        scale = self.target_size * 1.0 / max(original_size[0], original_size[1])
        newh, neww = original_size[0] * scale, original_size[1] * scale
        new_w = int(neww + 0.5)
        new_h = int(newh + 0.5)

        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = apply_coords(self,boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)