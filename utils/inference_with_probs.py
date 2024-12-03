import copy
import cv2
import torch.nn.functional as F
import os.path as osp
import glob
import numpy as np
import torch
from mmdet.apis import init_detector, inference_detector
from mmcv.parallel import collate, scatter
from mmcv.ops.nms import batched_nms
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmcv.ops import RoIPool

def my_multiclass_nms(multi_bboxes,
                      multi_scores,
                      score_thr,
                      nms_cfg,
                      max_num=-1, ):
    """NMS for multi-class bboxes.
    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
    Returns:
        tuple: (bboxes, labels, probs ), tensors of shape (k, 5),
            (k), and (k, 80). Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels.view(1, -1).expand_as(scores)

    # choose max score for each ROI
    _, indices = torch.max(scores, dim=1)

    device = scores.device
    num_ROIs = indices.size(0)
    new_bboxes = torch.zeros(num_ROIs, 4).to(device)
    new_scores = torch.zeros(num_ROIs, ).to(device)
    new_labels = torch.zeros(num_ROIs, ).to(device)
    for i in range(num_ROIs):
        idx = indices[i]
        new_bboxes[i, :] = bboxes[i, idx, :]
        new_scores[i] = scores[i, idx]
        new_labels[i] = labels[i, idx]

    # remove low scoring boxes
    valid_mask = new_scores > score_thr

    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    new_bboxes, new_scores, new_labels = new_bboxes[inds], new_scores[inds], new_labels[inds]

    probs = scores[inds]  # same as applying softmax operation on logits

    # If no valid detections are found, return empty tensors
    if new_bboxes.numel() == 0:
        empty_tensor = torch.empty((0,), device=device)
        return empty_tensor, empty_tensor, empty_tensor

    dets, keep = batched_nms(new_bboxes, new_scores, new_labels, nms_cfg)
    probs = probs[keep]
    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]
        probs = probs[:max_num]

    return dets, new_labels[keep], probs


def inference_detector_with_probs(model, img):
    """Inference image with the detector and return bounding boxes, labels, and probabilities.

    Args:
        model (nn.Module): The loaded detector.
        img (str/ndarray): Image file path or numpy array of the image.

    Returns:
        det_bboxes: List of detected bounding boxes.
        det_labels: List of detected labels for the bounding boxes.
        det_probs: List of probability scores for the detected labels.
    """

    # Check if model is wrapped in DistributedDataParallel
    if hasattr(model, 'module'):
        cfg = model.module.cfg
        model_ = model.module
    else:
        cfg = model.cfg
        model_ = model
    device = next(model.parameters()).device  # model device

    # Handle numpy array input
    if isinstance(img, np.ndarray):
        cfg = cfg.copy()
        # Update pipeline to handle ndarray
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    # Replace 'ImageToTensor' with 'DefaultFormatBundle'
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    # Prepare data
    if not isinstance(img, list):
        img = [img]

    datas = []
    for img_single in img:
        data = dict(img=img_single)
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=1)
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]

    # Scatter data to the correct device (GPU/CPU)
    data = scatter(data, [device])[0]

    img_metas = data["img_metas"][0]

    # Forward pass through the model
    with torch.no_grad():
        # Extract features
        x = model_.extract_feat(data['img'][0])
        # RPN forward
        proposal_list = model_.rpn_head.simple_test_rpn(x, img_metas)

        # Prepare rois for bbox head
        rois = torch.cat([p.unsqueeze(0) for p in proposal_list], dim=0)

        batch_index = torch.arange(
            rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(
            rois.size(0), rois.size(1), 1)
        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Flatten rois
        rois = rois.view(-1, 5)
        # BBox head forward
        bbox_results = model_.roi_head._bbox_forward(x, rois)
        cls_logits = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.view(batch_size, num_proposals_per_img, -1)
        cls_logits = cls_logits.view(batch_size, num_proposals_per_img, -1)
        bbox_pred = bbox_pred.view(batch_size, num_proposals_per_img, -1)

        # Apply softmax to get class probabilities
        scores = F.softmax(cls_logits, dim=-1)

        # Decode boxes
        img_shape = img_metas[0]['img_shape']
        num_classes = cls_logits.shape[-1] - 1  # Exclude background
        bboxes = model_.roi_head.bbox_head.bbox_coder.decode(
            rois[..., 1:], bbox_pred, max_shape=img_shape)

        # Reshape bboxes to [batch_size, num_proposals_per_img, num_classes, 4]
        bboxes = bboxes.view(batch_size, num_proposals_per_img, num_classes, 4)

        # Reshape scale_factor to [1, 1, 1, 4] for broadcasting
        scale_factor = bboxes.new_tensor(img_metas[0]['scale_factor']).view(1, 1, 1, 4)

        # Perform element-wise division
        bboxes /= scale_factor

        det_bboxes = []
        det_labels = []
        det_probs = []

        # Perform NMS using your my_multiclass_nms
        for i in range(batch_size):
            bbox = bboxes[i]  # [num_proposals_per_img, num_classes, 4]
            score = scores[i, :, :-1]  # Exclude background class

            # Reshape for NMS
            bbox = bbox.reshape(-1, 4 * num_classes)
            score = score.reshape(-1, num_classes)

            # Call your my_multiclass_nms function
            det_bbox, det_label, det_prob = my_multiclass_nms(
                bbox, score, cfg.model.test_cfg.rcnn.score_thr,
                cfg.model.test_cfg.rcnn.nms, cfg.model.test_cfg.rcnn.max_per_img)

            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            det_probs.append(det_prob)

    return det_bboxes, det_labels, det_probs
