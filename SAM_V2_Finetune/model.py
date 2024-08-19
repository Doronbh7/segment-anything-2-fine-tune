import torch.nn as nn
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os
import pickle
import numpy as np
from PIL.Image import Image


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.predictor=None

    def setup(self):
        self.model = build_sam2(self.cfg.model.type, self.cfg.model.base_model_checkpoint)
        self.predictor=SAM2ImagePredictor(self.model)
        if(self.cfg.model.Train_from_fine_tuned_model==True):
            self.predictor.model.load_state_dict(torch.load(self.cfg.model.fine_tuned_checkpoint))

        if self.cfg.model.freeze.prompt_encoder==False:
            self.predictor.model.sam_prompt_encoder.train(True)
        if self.cfg.model.freeze.mask_decoder==False:
            self.predictor.model.sam_mask_decoder.train(True)


    def forward(self, images,name, bboxes=None, centers=None,previous_masks=None):
        if not bboxes and not centers:
            raise ValueError("Either bboxes or centers must be provided")

        predictor=self.predictor

##image embedding cache store and load(reduce 35% of training time)
        if(self.cfg.save_image_embeddings==True):

            features_file_name=os.path.join(self.cfg.image_features_embeddings_dir,(str(name)+"_image_embeddings_cache.pklz"))

            if os.path.exists(features_file_name):
                with open(features_file_name, 'rb') as f:
                    image_features = pickle.load(f)

                    predictor.reset_predictor()
                    # Transform the image to the form expected by the model
                    if isinstance(images, np.ndarray):
                        predictor._orig_hw = [images.shape[:2]]
                    elif isinstance(images, Image):
                        w, h = images.size
                        predictor._orig_hw = [(h, w)]
                    else:
                        raise NotImplementedError("Image format not supported")
                    predictor._features = image_features
                    predictor._is_image_set = True
            else:

                predictor.set_image(images)
                image_features = predictor._features
                with open(features_file_name, 'wb') as f:
                    pickle.dump(image_features, f)
        else:
            predictor.set_image(images)

        pred_masks = []
        ious = []
        if not centers:
            centers = [None] * len(bboxes)
        if not bboxes:
            bboxes = [None] * len(centers)
        for  bbox, center in zip( bboxes, centers):
            if(self.cfg.prompt_type=="points"):
                mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(center[0], center[1], box=None,
                                                                                        mask_logits=previous_masks,
                                                                                        normalize_coords=True)
                if(mask_input is not None):
                    mask_input =mask_input.unsqueeze(0)
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),
                                                                                         boxes=None, masks=mask_input, )
                batched_mode = unnorm_coords.shape[0] > 1  # multi object prediction

            elif(self.cfg.prompt_type=="bounding_box"):
                mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(None,None, box=bbox,
                                                                                        mask_logits=previous_masks,
                                                                                        normalize_coords=True)
                if (mask_input is not None):
                    mask_input = mask_input.unsqueeze(0)

                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=None,
                                                                                         boxes=bbox, masks=mask_input, )
                batched_mode = unnorm_box.shape[0] > 1  # multi object prediction

            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=True,repeat_image=batched_mode,high_res_features=high_res_features,)
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])# Upscale the masks to the original image resolution

            pred_masks.append(prd_masks.squeeze(1))
            ious.append(prd_scores)

        return pred_masks, ious





