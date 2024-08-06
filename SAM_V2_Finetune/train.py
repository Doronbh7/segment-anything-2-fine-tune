import time
from torchvision.transforms.functional import resize, to_tensor,to_pil_image  # type: ignore

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
from utils import AverageMeter, calc_iou, best_score_mask,show_box,show_points



torch.set_float32_matmul_precision('high')

import os
import matplotlib.pyplot as plt
import numpy as np

def save_segmentation(images, pred_masks,pred_scores, gt_masks, name, centers,bboxes):
    """Function to save segmentation results as JPG files"""
    output_dir = cfg.segmentated_validation_images_dir
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'magenta', 'orange', 'lime', 'pink']

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    tensor_img=to_tensor(images)
    image_array = tensor_img.cpu().permute(1, 2, 0).numpy()

    axes[0].imshow(image_array)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    pred_overlay = image_array.copy()

    gt_overlay = image_array.copy()
    for mask_idx in range(pred_masks[0].size(0)):
        best_mask, score = best_score_mask(pred_masks[0][mask_idx].cpu(), pred_scores[0][mask_idx])
        prd_mask = best_mask.cpu().numpy()
        gt_mask = gt_masks[0][mask_idx].cpu().numpy()
        color = plt.get_cmap('tab10')(mask_idx % len(colors))

        pred_overlay[prd_mask > 0.5] = (1 - 0.5) * pred_overlay[prd_mask > 0.5] + 0.5 * np.array(color[:3])
        gt_overlay[gt_mask > 0.5] = (1 - 0.5) * gt_overlay[gt_mask > 0.5] + 0.5 * np.array(color[:3])

    axes[1].imshow(pred_overlay)
    input_label = np.array([1])

    if (cfg.prompt_type == "bounding_box"):
        for i in range(len(bboxes[0])):
            show_box(bboxes[0][i].cpu(), axes[1])
    if (cfg.prompt_type == "points"):
        show_points(centers[0][0].cpu().permute(1, 2, 0), input_label, axes[1])

    axes[1].set_title('Predicted Mask with the prompt')
    axes[1].axis('off')

    axes[2].imshow(pred_overlay)

    axes[2].set_title('Predicted Mask Overlay')
    axes[2].axis('off')

    axes[3].imshow(gt_overlay)
    axes[3].set_title('Ground Truth Mask Overlay')
    axes[3].axis('off')
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'{name[0]}.jpg')
    plt.savefig(filename)
    plt.close(fig)

def validate(fabric: L.Fabric, model: Model, val_dataloader: DataLoader, epoch: int=0):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks, name, centers = data

            if (cfg.prompt_type == "bounding_box"):
                pred_masks, pred_scores = model(images, name, bboxes=bboxes)
                for idx, (pred_mask,pred_score, gt_mask) in enumerate(zip(pred_masks[0],pred_scores[0], gt_masks[0])):
                    prd_mask,score=best_score_mask(pred_mask,pred_score)
                    prd_mask=prd_mask.cpu()
                    gt_mask_cpu=gt_mask.cpu()

                    batch_stats = smp.metrics.get_stats(
                        prd_mask,
                        gt_mask_cpu.int(),
                        mode='binary',
                        threshold=0.5,
                    )
                    batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                    batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                    ious.update(batch_iou, 1)
                    f1_scores.update(batch_f1, 1)

            if(cfg.prompt_type=="points"):
                pred_masks, pred_scores = model(images,name,centers=centers)

                for idx, (pred_mask,pred_score, gt_mask) in enumerate(zip(pred_masks[0],pred_scores[0], gt_masks[0])):
                    prd_mask,score=best_score_mask(pred_mask,pred_score)
                    prd_mask=prd_mask.cpu()
                    gt_mask_cpu=gt_mask.cpu()

                    batch_stats = smp.metrics.get_stats(
                        prd_mask,
                        gt_mask_cpu.int(),
                        mode='binary',
                        threshold=0.5,
                    )
                    batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                    batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                    ious.update(batch_iou, 1)
                    f1_scores.update(batch_f1, 1)




            # Save the segmentation for the images
            if(cfg.save_validation_images_result==True):
              save_segmentation(images, pred_masks,pred_scores, gt_masks,name,centers,bboxes)

            fabric.print(
                f'Val: [{epoch}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )

    fabric.print(f'Validation [{epoch}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')

    fabric.print(f"Saving checkpoint to {cfg.out_checkpoint_dir}")
    state_dict = model.model.state_dict()
    if fabric.global_rank == 0:
        torch.save(state_dict, os.path.join(cfg.out_checkpoint_dir, f"epoch-{epoch:06d}-f1{f1_scores.avg:.2f}-ckpt.pth"))
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
            if epoch == 1 and epoch % cfg.eval_interval == 0 and not validated:
                validate(fabric, model, val_dataloader, epoch)
                validated = True

            data_time.update(time.time() - end)
            images, bboxes, gt_masks,name, centers = data

            batch_size = 1

            if (cfg.prompt_type == "bounding_box"):
                pred_masks, iou_predictions = model(images, name, bboxes=bboxes)
            if (cfg.prompt_type == "points"):
                pred_masks, iou_predictions = model(images, name, centers=centers)
            num_masks = len(pred_masks[0])

            loss_focal = torch.tensor(0., device=fabric.device)
            loss_dice = torch.tensor(0., device=fabric.device)
            loss_iou = torch.tensor(0., device=fabric.device)
            for prd_mask, gt_mask, pred_score in zip(pred_masks[0], gt_masks[0], iou_predictions[0]):
                pred_mask, score = best_score_mask(prd_mask, pred_score)
                batch_iou = calc_iou(pred_mask, gt_mask)

                loss_focal += focal_loss(pred_mask, gt_mask)
                loss_dice += dice_loss(pred_mask, gt_mask)
                loss_iou += F.mse_loss(score, batch_iou, reduction='sum') / num_masks

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
                      loggers=[TensorBoardLogger(cfg.out_checkpoint_dir, name="lightning-sam")])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(cfg.out_checkpoint_dir, exist_ok=True)

    with fabric.device:
        model = Model(cfg)
        model.setup()

    train_data, val_data = load_datasets(cfg, cfg.dataset.image_size)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

    optimizer, scheduler = configure_opt(cfg, model)
    model, optimizer = fabric.setup(model, optimizer)

    train_sam(cfg, fabric, model, optimizer, scheduler, train_data, val_data)
    validate(fabric, model, val_data, epoch=cfg.num_epochs)


if __name__ == "__main__":
    main(cfg)
