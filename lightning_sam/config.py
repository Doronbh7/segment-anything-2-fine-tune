from box import Box

config = {
    "num_devices": 1,
    "batch_size": 2,
    "num_workers": 4,
    "num_epochs": 30,
    "eval_interval": 2,
    "out_dir": "out/training",
    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_h',
        "checkpoint":"/home/user_218/SAM_Project/SAM-ARMBench/lightning_sam/sam_vit_h_4b8939.pth",##/home/user_218/SAM_Project/FineTune_lightning/lightning-sam/lightning_sam/sam_vit_h_4b8939.pth",
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
