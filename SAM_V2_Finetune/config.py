from box import Box

config = {
    "num_devices": 1,
    "num_workers": 2,
    "num_epochs": 3,
    "eval_interval": 3,
    "out_checkpoint_dir": "out/training",
    "save_validation_images_result": False,
    "segmentated_validation_images_dir":"<segmentated_validation_images_dir path>",
    "prompt_type":"points", #points/bounding_box
    "save_image_embeddings":True,
    "save_embeddings_only_for_iterative_sampling": True,  # Temporarily save image embeddings only during the correction clicks loop. Useful for low storage.
    "image_features_embeddings_dir":"<image_features_embeddings_dir path>",
    "iterative_sampling":True,
    "correction_clicks":7,#(only for iterative sampling=True)
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
        "image_resize": 1024,
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
