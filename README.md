
# Segment-Anything-2-Finetune Project

Welcome to the Segment-Anything-2-Finetune project! This repository is designed to train and evaluate a segmentation model using Meta's Segment-Anything-2 and COCO format.

## Features
- **Dataset Configuration**: COCO format .
- **Training Options**: Train the model using either bounding boxes or points. Points are generated from the bounding boxes.Each point represent the center of the bounding box .
- **Mask Utilization**: Multiple masks are utilized for each point or bounding box, with the highest-scoring mask being used for training (multimask_output=True).
- **Efficient Training**: Save and load image embeddings to reduce training time. Save ~35% of training time by loading embeddings from a previous epoch/run.
- **Validation Output**: Save segmented validation images to a specified directory.


## How to Use
**1.Configure Settings**:
    Open the configuration file.
    Adjust the settings to match your desired parameters.

**2.Run Training**:
    Run the train.py file to start training.

Special thanks to [luca-medeiros](https://github.com/luca-medeiros) and [sagieppel](https://github.com/sagieppel).  Their code served as the base for this project.

## Resources

- [Luca Medeiros' Lightning SAM](https://github.com/luca-medeiros/lightning-sam)
- [Fine-Tune and Train Segment Anything 2 in 60 Lines of Code](https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code)
- [Lightning SAM Point Prompt](https://github.com/Garfield-hr/lightning-sam-point-prompt)
- [Segment Anything Model (SAM) Prompt Examples on Kaggle](https://www.kaggle.com/code/danpresil1/segment-anything-model-sam-prompt-examples)
- [Segment Anything 2 by Facebook Research](https://github.com/facebookresearch/segment-anything-2)
- [Lightning AI](https://github.com/Lightning-AI/lightning)

## License

This project is licensed under the same terms as the SAM 2 model.


## Citing SAM 2

@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}