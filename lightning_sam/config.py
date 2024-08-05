from box import Box

config = {
    "num_devices": 1,
    "num_workers": 2,
    "num_epochs": 5,
    "eval_interval": 5,
    "out_dir": "out/training",
    "segmentated_validation_images_dir":"<segmentated_validation_images_dir path>",
    "prompt_type":"points",#points/bounding_box/grid_prompt

    "opt": {
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'sam2_hiera_l.yaml', #sam2 model type (sam2_hiera_l.yaml,sam2_hiera_t.yaml,sam2_hiera_t.yaml,sam2_hiera_b+.yaml)
        "base_model_checkpoint": "<base model checkpoint path>",
        "Train_from_fine_tuned_model": False,
        "fine_tuned_checkpoint": "<fine tuned model check point path (Add path if you want to continue the training from the fine tuned model) >",
        "freeze": {
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": "<images path>",
            "annotation_file": "annotation file path (.json)"
        },
        "val": {
            "root_dir": "<images path>",
            "annotation_file": "annotation file path (.json)"
        }
    }
}

cfg = Box(config)
