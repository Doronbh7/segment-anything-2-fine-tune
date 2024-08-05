import os


from PIL import Image
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils import apply_boxes, apply_coords
class COCODataset(Dataset):

    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        # Filter out image_ids without any annotations
        self.image_ids = [image_id for image_id in self.image_ids if len(self.coco.getAnnIds(imgIds=image_id)) > 0]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        name=image_info['file_name'];
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        masks = []
        centers = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            bboxes.append([x, y, x + w, y + h])
            mask = self.coco.annToMask(ann)
            masks.append(mask)
            centers.append([[x + w / 2, y + h / 2]])

        if self.transform:
            image, masks, bboxes,name, centers = self.transform(image, masks, np.array(bboxes),name, np.array(centers))

        bboxes = np.stack(bboxes, axis=0)
        masks = np.stack(masks, axis=0)


        centers = np.stack(centers, axis=0)
        labels = np.ones((len(centers), 1))
        labels_torch = torch.as_tensor(labels, dtype=torch.int)

        return image, torch.tensor(bboxes), torch.tensor(masks).float(),name, (torch.tensor(centers), labels_torch)


def collate_fn(batch):
    images, bboxes, masks,name, centers = zip(*batch)
    temp=images[0]
    images=temp

    return images, bboxes, masks,name, centers


class ResizeAndPad:

    def __init__(self, target_size):
        self.target_size = target_size
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, masks, bboxes,name, coords):
        # Resize image and masks

        og_w, og_h = image.size
        scale = self.target_size * 1.0 / max(og_h, og_w)
        newh, neww = og_h * scale, og_w * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        image=resize(image, (newh, neww))


        new_masks = []
        for mask in masks:
            og_h, og_w = mask.shape

            scale = self.target_size * 1.0 / max(og_h, og_w)
            newh, neww = og_h * scale, og_w * scale
            neww = int(neww + 0.5)
            newh = int(newh + 0.5)
            mask = resize(to_pil_image(mask), (newh, neww))

            new_masks.append(mask)



        # Pad image and masks to form a square
        w,h = image.size

        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)

        masks = [transforms.Pad(padding)(mask) for mask in new_masks]

        # Adjust bounding boxes
        bboxes = apply_boxes(self,bboxes, (og_h, og_w))
        bboxes = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in bboxes]

        coords = apply_coords(self,coords, (og_h, og_w))
        coords[..., 0] += pad_w
        coords[..., 1] += pad_h

        return image, masks, bboxes,name, coords


def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    train = COCODataset(root_dir=cfg.dataset.train.root_dir,
                        annotation_file=cfg.dataset.train.annotation_file,
                        transform=transform)
    val = COCODataset(root_dir=cfg.dataset.val.root_dir,
                      annotation_file=cfg.dataset.val.annotation_file,
                      transform=transform)
    train_dataloader = DataLoader(train,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=cfg.num_workers,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(val,
                                batch_size=1,
                                shuffle=True,
                                num_workers=cfg.num_workers,
                                collate_fn=collate_fn)
    return train_dataloader, val_dataloader
