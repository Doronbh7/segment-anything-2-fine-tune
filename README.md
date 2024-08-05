## segment-anything-2-finetune Project

This project is designed to train and evaluate a segmentation model using the Meta's segment-anything-2. 

# Segment-Anything-2-Finetune Project

Welcome to the Segment-Anything-2-Finetune project! This repository is designed to train and evaluate a segmentation model using Meta's Segment-Anything-2 and COCO data.

## Features

- **Training Options**: Train the model using either bounding boxes or points. Points are generated to represent the center of each bounding box.
- **Mask Utilization**: Multiple masks are utilized for each point or bounding box, with the highest-scoring mask being used for training (multimask_output=True).
- **Efficient Training**: Save and load image embeddings to reduce training time.
- **Validation Output**: Save segmented validation images to a specified directory.


## How to Use
**1.Configure Settings**:
    Open the configuration file.
    Adjust the settings to match your desired parameters.

**2.Run Training**:
    Run the train.py file to start training.

Special thanks to [luca-medeiros](https://github.com/luca-medeiros). His code was the foundation for this project.

## Resources

- [Luca Medeiros' Lightning SAM](https://github.com/luca-medeiros/lightning-sam)
- [Fine-Tune and Train Segment Anything 2 in 60 Lines of Code](https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code)
- [Lightning SAM Point Prompt](https://github.com/Garfield-hr/lightning-sam-point-prompt)
- [Segment Anything Model (SAM) Prompt Examples on Kaggle](https://www.kaggle.com/code/danpresil1/segment-anything-model-sam-prompt-examples)
- [Segment Anything 2 by Facebook Research](https://github.com/facebookresearch/segment-anything-2)
- [Lightning AI](https://github.com/Lightning-AI/lightning)

## License

This project is licensed under the same terms as the SAM 2 model.