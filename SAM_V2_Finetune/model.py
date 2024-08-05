import torch.nn as nn
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.predictor=None

    def setup(self):
        self.model = build_sam2(self.cfg.model.type, self.cfg.model.base_model_checkpoint)
        self.predictor=SAM2ImagePredictor(self.model)
        if(self.cfg.Train_from_fine_tuned_model==True):
            self.predictor.model.load_state_dict(torch.load(self.cfg.model.fine_tuned_checkpoint))

        if self.cfg.model.freeze.prompt_encoder:
            self.predictor.model.sam_prompt_encoder.train(True)
        if self.cfg.model.freeze.mask_decoder:
            self.predictor.model.sam_mask_decoder.train(True)


    def forward(self, images,name, bboxes=None, centers=None):
        if not bboxes and not centers:
            raise ValueError("Either bboxes or centers must be provided")


##image embedding cache store and load(reduce 90% of training time),works the best with 1 batch
        file_name=os.path.join(self.cfg.image_embeddings_dir,(str(name)+"_image_embeddings_cache.pklz"))
        predictor=self.predictor


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
                                                                                        mask_logits=None,
                                                                                        normalize_coords=True)
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),
                                                                                         boxes=None, masks=None, )
                batched_mode = unnorm_coords.shape[0] > 1  # multi object prediction

            elif(self.cfg.prompt_type=="bounding_box"):
                mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(None,None, box=bbox,
                                                                                        mask_logits=None,
                                                                                        normalize_coords=True)
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=None,
                                                                                         boxes=bbox, masks=None, )
                batched_mode = unnorm_box.shape[0] > 1  # multi object prediction

            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=True,repeat_image=batched_mode,high_res_features=high_res_features,)
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])# Upscale the masks to the original image resolution


            pred_masks.append(prd_masks.squeeze(1))
            ious.append(prd_scores)

        return pred_masks, ious





