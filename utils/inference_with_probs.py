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
        img (str/ndarray/Tensor): Image file path, numpy array, or tensor of the image. 
        Note that when you want to not cut off the gradient, you can only use tensor as input.

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

    # Handle tensor input directly to preserve gradients
    if isinstance(img, torch.Tensor):
        # Process tensor input while preserving gradients
        return _process_tensor_input(model_, cfg, img, device)
    else:
        # Process numpy array or image file path
        return _process_numpy_input(model_, cfg, img, device)


def _process_tensor_input(model_, cfg, img, device):
    """Process tensor input to preserve gradients.
    
    Args:
        model_ (nn.Module): The unwrapped detector model.
        cfg (mmcv.Config): Model configuration.
        img (torch.Tensor): Tensor of shape [C, H, W] or [B, C, H, W].
        device (torch.device): Device for computation.
        
    Returns:
        Tuple of (det_bboxes, det_labels, det_probs).
    """
    # Ensure input has correct shape [B, C, H, W]
    if len(img.shape) == 3:  # [C, H, W]
        img = img.unsqueeze(0)  # Convert to [1, C, H, W]
    elif len(img.shape) == 4:  # Already [B, C, H, W]
        pass
    else:
        raise ValueError(f"Unsupported tensor shape: {img.shape}")

    # Move tensor to correct device
    img = img.to(device)
    
    # Forward computation with or without gradients
    with torch.set_grad_enabled(img.requires_grad):
        # Extract features
        x = model_.extract_feat(img)
        
        # RPN forward pass
        img_shape = (img.shape[2], img.shape[3])
        proposal_list = model_.rpn_head.simple_test_rpn(x, [{'img_shape': img_shape}])
        
        # Process results
        return _process_model_outputs(model_, cfg, x, proposal_list, img_shape)


def _process_numpy_input(model_, cfg, img, device):
    """Process numpy array or file path input.
    
    Args:
        model_ (nn.Module): The unwrapped detector model.
        cfg (mmcv.Config): Model configuration.
        img (str/ndarray): Image file path or numpy array.
        device (torch.device): Device for computation.
        
    Returns:
        Tuple of (det_bboxes, det_labels, det_probs).
    """
    # Handle numpy array input, without grad
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
        
        # Get image shape and scale factor from img_metas
        img_shape = img_metas[0]['img_shape']
        scale_factor = img_metas[0]['scale_factor']
        
        # Process results
        return _process_model_outputs(model_, cfg, x, proposal_list, img_shape, scale_factor)


def _process_model_outputs(model_, cfg, x, proposal_list, img_shape, scale_factor=None, score_thr=0.5, sharpness=50.0):
    """
    Process model outputs to get differentiable bounding boxes, class scores, and logits.

    Args:
        model_ (nn.Module): The detector model.
        cfg (mmcv.Config): Model configuration.
        x (list): Feature maps from backbone.
        proposal_list (list): RPN proposals.
        img_shape (tuple): Image shape.
        scale_factor (tuple, optional): For resizing bbox back to original scale.
        score_thr (float): Confidence threshold for soft gating.
        sharpness (float): Controls the softness of gating (higher = sharper cutoff).

    Returns:
        bboxes: Tensor of shape [B, N, 4]
        scores: Tensor of shape [B, N, C] (probabilities, no background)
        cls_logits: Tensor of shape [B, N, C] (logits, no background)
    """
    # Construct ROI tensor
    rois = torch.cat([p.unsqueeze(0) for p in proposal_list], dim=0)
    batch_index = torch.arange(rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(rois.size(0), rois.size(1), 1)
    rois = torch.cat([batch_index, rois[..., :4]], dim=-1)  # [B, N, 5]
    batch_size, num_proposals_per_img = rois.shape[:2]

    # Flatten for ROI head
    rois = rois.view(-1, 5)
    bbox_results = model_.roi_head._bbox_forward(x, rois)
    cls_logits = bbox_results['cls_score']         # [B*N, C+1]
    bbox_pred = bbox_results['bbox_pred']          # [B*N, 4*(C)]

    # Restore batch shape
    rois = rois.view(batch_size, num_proposals_per_img, -1)
    cls_logits = cls_logits.view(batch_size, num_proposals_per_img, -1)  # [B, N, C+1]
    bbox_pred = bbox_pred.view(batch_size, num_proposals_per_img, -1)

    # Classification: softmax and extract foreground
    scores = F.softmax(cls_logits, dim=-1)          # [B, N, C+1]
    scores_no_bg = scores[:, :, :-1]                # [B, N, C]
    cls_logits_no_bg = cls_logits[:, :, :-1]        # [B, N, C]
    num_classes = scores_no_bg.shape[-1]

    # Decode per-class bboxes
    decoded_bboxes = model_.roi_head.bbox_head.bbox_coder.decode(
        rois[..., 1:], bbox_pred, max_shape=img_shape
    )  # [B, N, C*4]
    decoded_bboxes = decoded_bboxes.view(batch_size, num_proposals_per_img, num_classes, 4)

    if scale_factor is not None:
        scale_factor = decoded_bboxes.new_tensor(scale_factor).view(1, 1, 1, 4)
        decoded_bboxes /= scale_factor

    # Soft-argmax: compute weighted sum over all class predictions
    weights = scores_no_bg / (scores_no_bg.sum(dim=-1, keepdim=True) + 1e-6)  # [B, N, C]
    bboxes = torch.sum(decoded_bboxes * weights.unsqueeze(-1), dim=2)         # [B, N, 4]

    # Soft gating: suppress low-confidence proposals smoothly
    max_scores, _ = scores_no_bg.max(dim=-1)                                   # [B, N]
    soft_mask = torch.sigmoid((max_scores - score_thr) * sharpness)           # [B, N]

    bboxes = bboxes * soft_mask.unsqueeze(-1)                  # [B, N, 4]
    scores_no_bg = scores_no_bg * soft_mask.unsqueeze(-1)      # [B, N, C]
    cls_logits_no_bg = cls_logits_no_bg * soft_mask.unsqueeze(-1)  # [B, N, C]
    labels = torch.argmax(scores_no_bg, dim=-1)

    return bboxes, labels, cls_logits_no_bg

