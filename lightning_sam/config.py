from box import Box

config = {
    "num_devices": 1,
    "batch_size": 1,
    "num_workers": 4,
    "num_epochs": 0,
    "eval_interval": 1,
    "out_dir": "out/training",
    "image_embeddings_dir":"/home/user_218/SAM_Project/SAM-ARMBench/lightning_sam/Image_embeddings",
    "segmentated_validation_images_dir":"/home/user_218/SAM_Project/SAM-ARMBench/lightning_sam/segmentation_results",
    "opt": {
        "learning_rate": 1,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_h',
        "checkpoint":"/home/user_218/SAM_Project/SAM-ARMBench/lightning_sam/out/training/epoch-000100-f10.91-ckpt.pth",
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
