import torch
from torch import Tensor
from torchvision.ops import boxes as box_ops
import torch.nn.functional as F

class Matcher:
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        "BELOW_LOW_THRESHOLD": int,
        "BETWEEN_THRESHOLDS": int,
    }

    def __init__(self, high_threshold: float, low_threshold: float, allow_low_quality_matches: bool = False) -> None:
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        torch._assert(low_threshold <= high_threshold, "low_threshold should be <= high_threshold")
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix: Tensor) -> Tensor:
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError("No ground-truth boxes available for one of the images during training")
            else:
                raise ValueError("No proposal boxes available for one of the images during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None  # type: ignore[assignment]

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            if all_matches is None:
                torch._assert(False, "all_matches should not be None")
            else:
                self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches: Tensor, all_matches: Tensor, match_quality_matrix: Tensor) -> None:
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has the highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find the highest quality match available, even if it is low, including ties
        gt_pred_pairs_of_highest_quality = torch.where(match_quality_matrix == highest_quality_foreach_gt[:, None])
        # Example gt_pred_pairs_of_highest_quality:
        # (tensor([0, 1, 1, 2, 2, 3, 3, 4, 5, 5]),
        #  tensor([39796, 32055, 32070, 39190, 40255, 40390, 41455, 45470, 45325, 46390]))
        # Each element in the first tensor is a gt index, and each element in second tensor is a prediction index
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
 'banner', 'blanket', 'branch', 'bridge', 'building-other', 'bush',
 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain', 'desk-stuff',
 'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other', 'floor-stone',
 'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other', 'fruit',
 'furniture-other', 'grass', 'gravel', 'ground-other', 'hill', 'house',
 'leaves', 'light', 'mat', 'metal', 'mirror-stuff', 'moss', 'mountain', 'mud',
 'napkin', 'net', 'paper', 'pavement', 'pillow', 'plant-other', 'plastic',
 'platform', 'playingfield', 'railing', 'railroad', 'river', 'road', 'rock',
 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper',
 'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other', 'table',
 'tent', 'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick', 'wall-concrete',
 'wall-other', 'wall-panel', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other',
 'waterdrops', 'window-blind', 'window-other', 'wood', 'other']


nuimages_classes = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
                    ]


# def calculate_box_loss(boxes_pred, labels_pred, scores_pred, boxes, args):
#     batch_size = len(boxes_pred)
#
#     pred_boxes = torch.zeros((batch_size, args.max_boxes_per_img, 4), dtype=torch.float32, device=boxes.device)
#     pred_labels = torch.zeros((batch_size, args.max_boxes_per_img), dtype=torch.int64, device=boxes.device)
#     pred_scores = torch.zeros((batch_size, args.max_boxes_per_img), dtype=torch.float32, device=boxes.device)
#     pred_masks = torch.zeros((batch_size, args.max_boxes_per_img), dtype=torch.bool, device=boxes.device)
#
#     # Prepare ground truth data
#     gt_boxes = boxes[:, :, :4]
#     gt_labels = boxes[:, :, 4].long()
#
#     # matched_idxs = []
#     assigned_gt_boxes = []
#     assigned_gt_ids = []
#
#     for batch_id in range(batch_size):
#
#         # prevent detected objects > self.max_boxes_per_data
#         current_batch_size = min(len(boxes_pred[batch_id]), args.max_boxes_per_img)
#         for object_id in range(current_batch_size):
#             obj = torch.tensor(boxes_pred[batch_id][object_id], dtype=torch.float32).to(boxes.device)
#             label = torch.tensor(labels_pred[batch_id][object_id], dtype=torch.float32).to(boxes.device)
#             score = torch.tensor(scores_pred[batch_id][object_id], dtype=torch.float32).to(boxes.device)
#             pred_boxes[batch_id, object_id] = obj.clone().detach().to(boxes.device)
#             pred_labels[batch_id, object_id] = label.clone().detach().to(boxes.device)
#             pred_scores[batch_id, object_id] = score.clone().detach().to(boxes.device)
#             pred_masks[batch_id, object_id] = True
#
#         # Compute IoU between ground truth and predictions
#         match_quality_matrix = box_ops.box_iou(gt_boxes[batch_id], pred_boxes[batch_id])
#
#         proposal_matcher = Matcher(0.5, 0.5, allow_low_quality_matches=False)
#         matched_idxs_in_image = proposal_matcher(match_quality_matrix)
#
#         clamped_matched_idxs = matched_idxs_in_image.clamp(min=0)
#         assigned_gt_boxes.append(gt_boxes[batch_id][clamped_matched_idxs])
#         assigned_gt_ids.append(gt_labels[batch_id][clamped_matched_idxs])
#
#     assigned_gt_boxes = torch.stack(assigned_gt_boxes)
#     assigned_gt_ids = torch.stack(assigned_gt_ids)
#
#     flat_pred_labels = pred_labels[pred_masks].view(-1)
#     flat_assigned_gt_labels = assigned_gt_ids[pred_masks]
#     flat_scores = pred_scores[pred_masks].view(-1)
#
#     margin = 1.0
#     labels_one_hot = F.one_hot(flat_assigned_gt_labels, num_classes=args.num_classes).float()
#     flat_pred_labels_one_hot = F.one_hot(flat_pred_labels, num_classes=args.num_classes).float()
#
#     pred_boxes.requires_grad_()
#     flat_pred_labels_one_hot.requires_grad_()
#
#     loss_cls = torch.clamp(margin - flat_pred_labels_one_hot * labels_one_hot, min=0)
#     loss_cls = loss_cls * flat_scores.view(-1, 1)
#
#     # Compute losses
#     loss_box = F.l1_loss(pred_boxes[pred_masks==1], assigned_gt_boxes[pred_masks==1], reduction='none')
#     loss_box = loss_box * flat_scores.view(-1, 1)
#
#     return loss_box, loss_cls

# import torch
# import torch.nn.functional as F
#
# def calculate_box_loss(boxes_pred, labels_pred, scores_pred, boxes, args):
#     batch_size = len(boxes_pred)
#
#     pred_boxes = torch.zeros((batch_size, args.max_boxes_per_img, 4), dtype=torch.float32, device=boxes.device)
#     pred_labels = torch.zeros((batch_size, args.max_boxes_per_img), dtype=torch.int64, device=boxes.device)
#     pred_scores = torch.zeros((batch_size, args.max_boxes_per_img), dtype=torch.float32, device=boxes.device)
#     pred_masks = torch.zeros((batch_size, args.max_boxes_per_img), dtype=torch.bool, device=boxes.device)
#
#     # Prepare ground truth data
#     gt_boxes = boxes[:, :, :4]
#     gt_labels = boxes[:, :, 4].long()
#
#     assigned_gt_boxes = []
#     assigned_gt_ids = []
#
#     for batch_id in range(batch_size):
#         # Prevent detected objects > self.max_boxes_per_img
#         current_batch_size = min(len(boxes_pred[batch_id]), args.max_boxes_per_img)
#         for object_id in range(current_batch_size):
#             obj = torch.tensor(boxes_pred[batch_id][object_id], dtype=torch.float32).to(boxes.device)
#             label = torch.tensor(labels_pred[batch_id][object_id], dtype=torch.float32).to(boxes.device)
#             score = torch.tensor(scores_pred[batch_id][object_id], dtype=torch.float32).to(boxes.device)
#             pred_boxes[batch_id, object_id] = obj.clone().detach()
#             pred_labels[batch_id, object_id] = label.clone().detach()
#             pred_scores[batch_id, object_id] = score.clone().detach()
#             pred_masks[batch_id, object_id] = True
#
#
#         # Compute IoU between ground truth and predictions
#         match_quality_matrix = box_ops.box_iou(gt_boxes[batch_id], pred_boxes[batch_id])
#
#         proposal_matcher = Matcher(0.5, 0.5, allow_low_quality_matches=False)
#         matched_idxs_in_image = proposal_matcher(match_quality_matrix)
#
#         clamped_matched_idxs = matched_idxs_in_image.clamp(min=0)
#         assigned_gt_boxes.append(gt_boxes[batch_id][clamped_matched_idxs])
#         assigned_gt_ids.append(gt_labels[batch_id][clamped_matched_idxs])
#
#     assigned_gt_boxes = torch.stack(assigned_gt_boxes)
#     assigned_gt_ids = torch.stack(assigned_gt_ids)
#
#     # Now process all batches at once
#     loss_box_list = []
#     loss_cls_list = []
#
#     for batch_id in range(batch_size):
#         pred_boxes_single = pred_boxes[batch_id][pred_masks[batch_id]]
#         assigned_gt_boxes_single = assigned_gt_boxes[batch_id][pred_masks[batch_id]]
#         pred_labels_single = pred_labels[batch_id][pred_masks[batch_id]]
#         assigned_gt_labels_single = assigned_gt_ids[batch_id][pred_masks[batch_id]]
#         pred_scores_single = pred_scores[batch_id][pred_masks[batch_id]]
#
#         # Classification loss
#         margin = 1.0
#         labels_one_hot = F.one_hot(assigned_gt_labels_single, num_classes=args.num_classes).float()
#         pred_labels_one_hot = F.one_hot(pred_labels_single, num_classes=args.num_classes).float()
#
#         loss_cls_single = torch.clamp(margin - pred_labels_one_hot * labels_one_hot, min=0)
#         loss_cls_single = (loss_cls_single * pred_scores_single.view(-1, 1)).sum(dim=1).mean()
#
#         # Bounding box loss
#         loss_box_single = F.l1_loss(pred_boxes_single, assigned_gt_boxes_single, reduction='none')
#         loss_box_single = (loss_box_single * pred_scores_single.view(-1, 1)).sum(dim=1).mean()
#
#         loss_box_list.append(loss_box_single)
#         loss_cls_list.append(loss_cls_single)
#
#     loss_box = torch.stack(loss_box_list).view(batch_size, 1)
#     loss_cls = torch.stack(loss_cls_list).view(batch_size, 1)
#
#     return loss_box, loss_cls

import torch
import torch.nn.functional as F

def calculate_box_loss(boxes_pred, labels_pred, logits_pred, boxes, args):
    batch_size = len(boxes_pred)

    loss_box_total = 0.0
    loss_cls_total = 0.0

    # Prepare ground truth data
    gt_boxes = boxes[:, :, :4]  # (bs, max_boxes, 4)
    gt_labels = boxes[:, :, 4].long()  # (bs, max_boxes)

    for batch_id in range(batch_size):
        # Check if this batch should be considered for loss calculation

        # Get the current batch's predicted boxes, labels, and logits
        pred_boxes_single = boxes_pred[batch_id]
        pred_labels_single = labels_pred[batch_id]
        logits_pred_single = logits_pred[batch_id]

        if len(pred_boxes_single) == 0:  # No predicted boxes
            # Add penalty loss for empty predictions (can be adjusted based on your requirements)
            loss_cls_total += torch.tensor(1.0, device=boxes.device)
            loss_box_total += torch.tensor(1.0, device=boxes.device)
            continue

        # Convert to tensors and move to the correct device
        pred_boxes_single = torch.tensor(pred_boxes_single, dtype=torch.float32, device=boxes.device)
        pred_labels_single = torch.tensor(pred_labels_single, dtype=torch.long, device=boxes.device)
        logits_pred_single = torch.tensor(logits_pred_single, dtype=torch.float32, device=boxes.device)

        # Compute IoU between ground truth and predictions
        match_quality_matrix = box_ops.box_iou(gt_boxes[batch_id], pred_boxes_single)

        # Assign ground truth to predictions
        proposal_matcher = Matcher(0.5, 0.5, allow_low_quality_matches=False)
        matched_idxs_in_image = proposal_matcher(match_quality_matrix)
        clamped_matched_idxs = matched_idxs_in_image.clamp(min=0)

        # Get the corresponding ground truth boxes and labels
        assigned_gt_boxes = gt_boxes[batch_id][clamped_matched_idxs]
        assigned_gt_labels = gt_labels[batch_id][clamped_matched_idxs]

        # Classification loss using logits and cross-entropy
        loss_cls_single = F.cross_entropy(logits_pred_single / 0.1, assigned_gt_labels, reduction='mean')

        # Bounding box loss (L1 loss)
        loss_box_single = F.l1_loss(pred_boxes_single, assigned_gt_boxes, reduction='mean')

        # Accumulate losses
        loss_cls_total += loss_cls_single
        loss_box_total += loss_box_single

    return loss_box_total, loss_cls_total



import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_predictions(image, boxes_score_pred_image, labels_pred_image, gt_boxes, score_threshold=0.5):
    # 创建一个Figure
    fig, ax = plt.subplots(1)
    # 将原始图像放在背景
    ax.imshow(image)

    # 遍历每个预测的box
    for i, box in enumerate(boxes_score_pred_image):
        x1, y1, x2, y2, score = box
        label = labels_pred_image[i]

        # 只绘制 score 大于阈值的框
        if score > score_threshold:
            # 创建矩形框 (红色框为预测框)
            width, height = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')

            # 绘制矩形框
            ax.add_patch(rect)

            # 在框的左上角显示标签和得分
            ax.text(x1, y1 - 5, f'Label: {label}, Score: {score:.2f}', color='yellow', fontsize=12, weight='bold', bbox=dict(facecolor='blue', alpha=0.5))

    # 遍历每个gt box
    for i, box in enumerate(gt_boxes):
        x1, y1, x2, y2, label = box

        # 跳过全零的box
        if torch.all(box == 0):
            continue

        # 创建矩形框 (绿色框为真实框)
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1 * image.shape[1], y1 * image.shape[0]), width * image.shape[1], height * image.shape[0],
                                 linewidth=2, edgecolor='g', facecolor='none')

        # 绘制矩形框
        ax.add_patch(rect)

        # 在框的左上角显示标签 (真实框的标签)
        ax.text(x1 * image.shape[1], (y1 * image.shape[0]) - 5, f'GT Label: {int(label)}', color='white', fontsize=12, weight='bold', bbox=dict(facecolor='green', alpha=0.5))

    plt.axis('off')
    plt.show()

# 示例调用
# visualize_predictions(image, boxes_score_pred_image, labels_pred_image, gt_boxes)
# tensor([ 66, 166, 283, 413,  55, 360, 168, 310, 268,  73, 299, 616, 681, 163,
#          754, 520], device='cuda:0')
# tensor([100, 407, 952, 299, 866, 901, 589, 801, 231,   5,  20, 527,  53, 441,
#         609, 329], device='cuda:0')
# tensor([100, 407, 952, 299, 866, 901, 589, 801, 231,   5,  20, 527,  53, 441,
#         609, 329], device='cuda:0')