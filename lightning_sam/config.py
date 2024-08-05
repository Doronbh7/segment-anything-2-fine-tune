from box import Box

config = {
    "num_devices": 1,
    "batch_size": 1,
    "num_workers": 2,
    "num_epochs": 1,
    "eval_interval": 1,
    "out_dir": "out/training",
    "image_embeddings_dir":"/home/user_218/SAM_Project/SAM-ARMBench/lightning_sam/mobile_sam_embedding",
    "segmentated_validation_images_dir":"/home/user_218/SAM_Project/SAM-ARMBench/lightning_sam/segmentation_results",
    "prompt_type":"points",#points/bounding_box/grid_prompt (only show image output for grid prompt with validation set,use with eval interval 1 and epoch 1)

    "opt": {
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'sam2_hiera_l.yaml', #mobile_sam - vit_t / regular vit_h
        "base_model_checkpoint": "/home/user_218/SAM_Project/SAM-ARMBench/segment_anything_2/checkpoints/sam2_hiera_large.pt", #"/home/user_218/SAM_Project/SAM-ARMBench/MobileSam/weights/mobile_sam.pt",
        "fine_tuned_checkpoint": "/home/user_218/SAM_Project/SAM-ARMBench/lightning_sam/out/training/epoch-000000-f10.97-ckpt.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": "/home/user_218/SAM_Project/SAM-ARMBench/content/armbench-segmentation-0.1/zoomed-out-tote-transfer-set/images",
            "annotation_file": "/home/user_218/SAM_Project/SAM-ARMBench/content/armbench-segmentation-0.1/zoomed-out-tote-transfer-set/train.json"
        },
        "val": {
                "root_dir": "/home/user_218/SAM_Project/SAM-ARMBench/content/armbench-segmentation-0.1/zoomed-out-tote-transfer-set/images",
            "annotation_file": "/home/user_218/SAM_Project/SAM-ARMBench/content/armbench-segmentation-0.1/zoomed-out-tote-transfer-set/val.json"
        }
    }
}

cfg = Box(config)
