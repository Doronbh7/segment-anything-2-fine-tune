import os
import time
import matplotlib.pyplot as plt

import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from box import Box
from config import cfg
from dataset import load_datasets
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
from losses import DiceLoss
from losses import FocalLoss
from model import Model
from torch.utils.data import DataLoader
from utils import AverageMeter
from utils import calc_iou


torch.set_float32_matmul_precision('high')

import os
import matplotlib.pyplot as plt
import numpy as np


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


def save_segmentation(images, pred_masks, gt_masks, name, centers,bboxes):
    """Function to save segmentation results as JPG files"""
    output_dir = cfg.segmentated_validation_images_dir
    batch_size = images.size(0)
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'magenta', 'orange', 'lime', 'pink']

    for idx in range(batch_size):
        if(cfg.prompt_type == "grid_prompt"):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        else:
            fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        axes[0].imshow(images[idx].cpu().permute(1, 2, 0))
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        pred_overlay = images[idx].cpu().permute(1, 2, 0).numpy().copy()

        if (cfg.prompt_type != "grid_prompt"):
            gt_overlay = images[idx].cpu().permute(1, 2, 0).numpy().copy()

        for mask_idx in range(pred_masks[idx].size(0)):
            pred_mask = pred_masks[idx][mask_idx].cpu().numpy()
            if (cfg.prompt_type != "grid_prompt"):
                gt_mask = gt_masks[idx][mask_idx].cpu().numpy()
            color = plt.get_cmap('tab10')(mask_idx % len(colors))

            pred_overlay[pred_mask > 0.5] = (1 - 0.5) * pred_overlay[pred_mask > 0.5] + 0.5 * np.array(color[:3])
            if (cfg.prompt_type != "grid_prompt"):
                gt_overlay[gt_mask > 0.5] = (1 - 0.5) * gt_overlay[gt_mask > 0.5] + 0.5 * np.array(color[:3])

        axes[1].imshow(pred_overlay)
        input_label = np.array([1])

        if (cfg.prompt_type == "bounding_box"):
            for i in range(len(bboxes[idx])):
                show_box(bboxes[idx][i].cpu(), axes[1])
        if (cfg.prompt_type == "points"):
            show_points(centers[idx][0].cpu().permute(1, 2, 0), input_label, axes[1])
        if (cfg.prompt_type == "grid_prompt"):
            show_points(centers[idx][0].cpu().permute(1, 2, 0), input_label, axes[1])

        axes[1].set_title('Predicted Mask with the prompt')
        axes[1].axis('off')

        axes[2].imshow(pred_overlay)


        axes[2].set_title('Predicted Mask Overlay')
        axes[2].axis('off')

        if(cfg.prompt_type != "grid_prompt"):
            axes[3].imshow(gt_overlay)
            axes[3].set_title('Ground Truth Mask Overlay')
            axes[3].axis('off')

        # Save the figure
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f'{name[idx]}.jpg')
        plt.savefig(filename)
        plt.close(fig)

def validate(fabric: L.Fabric, model: Model, val_dataloader: DataLoader, epoch: int = 0):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks, name, centers = data
            num_images = images.size(0)

            if (cfg.prompt_type == "bounding_box"):
                pred_masks, _ = model(images, name, bboxes=bboxes)
                for idx, (pred_mask, gt_mask) in enumerate(zip(pred_masks, gt_masks)):
                    batch_stats = smp.metrics.get_stats(
                        pred_mask,
                        gt_mask.int(),
                        mode='binary',
                        threshold=0.5,
                    )
                    batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                    batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                    ious.update(batch_iou, num_images)
                    f1_scores.update(batch_f1, num_images)

            if(cfg.prompt_type=="points"):
                pred_masks, _ = model(images,name,centers=centers)
                for idx, (pred_mask, gt_mask) in enumerate(zip(pred_masks, gt_masks)):
                    batch_stats = smp.metrics.get_stats(
                        pred_mask,
                        gt_mask.int(),
                        mode='binary',
                        threshold=0.5,
                    )
                    batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                    batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                    ious.update(batch_iou, num_images)
                    f1_scores.update(batch_f1, num_images)

            if (cfg.prompt_type == "grid_prompt"):
                pred_masks, _ = model(images, name, centers=centers)

            # Save the segmentation for the images
            save_segmentation(images, pred_masks, gt_masks,name,centers,bboxes)

            fabric.print(
                f'Val: [{epoch}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )

    fabric.print(f'Validation [{epoch}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')

    fabric.print(f"Saving checkpoint to {cfg.out_dir}")
    state_dict = model.model.state_dict()
    if fabric.global_rank == 0:
        torch.save(state_dict, os.path.join(cfg.out_dir, f"epoch-{epoch:06d}-f1{f1_scores.avg:.2f}-ckpt.pth"))
    model.train()


def train_sam(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    """The SAM training loop."""

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()

    for epoch in range(1, cfg.num_epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()
        end = time.time()
        validated = False

        for iter, data in enumerate(train_dataloader):
            if epoch > 1 and epoch % cfg.eval_interval == 0 and not validated:
                validate(fabric, model, val_dataloader, epoch)
                validated = True

            data_time.update(time.time() - end)
            images, bboxes, gt_masks,name, centers = data
            batch_size = images.size(0)

            if (cfg.prompt_type == "bounding_box"):
                pred_masks, iou_predictions = model(images, name, bboxes=bboxes)
            if (cfg.prompt_type == "points"):
                pred_masks, iou_predictions = model(images, name, centers=centers)
            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_focal = torch.tensor(0., device=fabric.device)
            loss_dice = torch.tensor(0., device=fabric.device)
            loss_iou = torch.tensor(0., device=fabric.device)
            for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, iou_predictions):
                batch_iou = calc_iou(pred_mask, gt_mask)
                loss_focal += focal_loss(pred_mask, gt_mask)
                loss_dice += dice_loss(pred_mask, gt_mask)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks

            loss_total = 20. * loss_focal + loss_dice + loss_iou
            optimizer.zero_grad()
            fabric.backward(loss_total)
            optimizer.step()
            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

            focal_losses.update(loss_focal.item(), batch_size)
            dice_losses.update(loss_dice.item(), batch_size)
            iou_losses.update(loss_iou.item(), batch_size)
            total_losses.update(loss_total.item(), batch_size)

            fabric.print(f'Epoch: [{epoch}][{iter+1}/{len(train_dataloader)}]'
                         f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                         f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                         f' | a Focal Loss [{20. * focal_losses.val:.4f} ({20. * focal_losses.avg:.4f})]'                         f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
                         f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
                         f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')
            steps = epoch * len(train_dataloader) + iter
            log_info = {
                'Loss': total_losses.val,
                'alpha focal loss': 20. * focal_losses.val,
                'dice loss': dice_losses.val,
            }
            fabric.log_dict(log_info, step=steps)

def configure_opt(cfg: Box, model: Model):

    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def main(cfg: Box) -> None:
    fabric = L.Fabric(accelerator="auto",
                      devices=cfg.num_devices,
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir, name="lightning-sam")])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(cfg.out_dir, exist_ok=True)

    with fabric.device:
        model = Model(cfg)
        model.setup()

    train_data, val_data = load_datasets(cfg, model.model.image_encoder.img_size)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

    optimizer, scheduler = configure_opt(cfg, model)
    model, optimizer = fabric.setup(model, optimizer)

    train_sam(cfg, fabric, model, optimizer, scheduler, train_data, val_data)
    validate(fabric, model, val_data, epoch=0)


if __name__ == "__main__":
    main(cfg)
