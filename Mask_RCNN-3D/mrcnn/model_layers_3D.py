import numpy as np
import tensorflow as tf
import keras
import keras.engine as KE
from keras.models import Model
from keras.layers import Conv3D, Conv3DTranspose, Dense, Reshape
from keras.layers import Input, Lambda, Activation, TimeDistributed

import utils_3D as utils
from utils_3D import non_max_suppression as nms_3D
import model_utils_3D as mutils
from model_utils_3D import BatchNorm, batch_slice
import crop_resize_3D
from crop_resize_3D import crop_and_resize as cr_3D

# from nms_3D import nms_3D

############################################################
#  Proposal Layer
############################################################

def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (z1, y1, x1, z2, y2, x2)] boxes to update
    deltas: [N, (dz, dy, dx, log(dd), log(dh), log(dw))] refinements to apply
    """
    # Convert to z, y, x, d, h, w
    depth  = boxes[:, 3] - boxes[:, 0]
    height = boxes[:, 4] - boxes[:, 1]
    width  = boxes[:, 5] - boxes[:, 2]
    center_z = boxes[:, 0] + 0.5 * depth
    center_y = boxes[:, 1] + 0.5 * height
    center_x = boxes[:, 2] + 0.5 * width
    # Apply deltas
    center_z += deltas[:, 0] * depth
    center_y += deltas[:, 1] * height
    center_x += deltas[:, 2] * width
    depth *= tf.exp(deltas[:, 3])
    height *= tf.exp(deltas[:, 4])
    width *= tf.exp(deltas[:, 5])
    # Convert back to z1, y1, x1, z2, y2, x2
    z1 = center_z - 0.5 * depth
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    z2 = z1 + depth
    y2 = y1 + height
    x2 = x1 + width
    return tf.stack([z1, y1, x1, z2, y2, x2], axis=1, name="apply_box_deltas_out")


def clip_boxes(boxes, window):
    """
    boxes: [N, (z1, y1, x1, z2, y2, x2)]
    window: [6] in the form z1, y1, x1, z2, y2, x2
    """
    # Split
    wz1, wy1, wx1, wz2, wy2, wx2 = tf.split(window, 6)
    z1, y1, x1, z2, y2, x2 = tf.split(boxes, 6, axis=1)
    # Clip
    z1 = tf.maximum(tf.minimum(z1, wz2), wz1)
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    z1 = tf.maximum(tf.minimum(z2, wz2), wz1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([z1, y1, x1, z2, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 6))
    return clipped


def nms_proposals(boxes, scores, proposal_count, nms_threshold):
     indices = mutils.non_max_suppression(boxes, scores, proposal_count, nms_threshold)
     proposals = tf.gather(boxes, indices)
     # Pad if needed
     padding = tf.maximum(proposal_count - tf.shape(proposals)[0], 0)
     proposals = tf.pad(proposals, [(0, padding), (0, 0)])
     return proposals


class ProposalLayer(KE.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.
    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dz, dy, dx, log(dd), log(dh), log(dw))]
        anchors: [batch, num_anchors, (z1, y1, x1, z2, y2, x2)] anchors in normalized coords
    Returns:
        Proposals in normalized coordinates [batch, rois, (z1, y1, x1, z2, y2, x2)]
    """

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 6]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 6])
        # Anchors
        anchors = inputs[2]

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices
        scores = batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        deltas = batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        pre_nms_anchors = batch_slice([anchors, ix],
                                      lambda a, x: tf.gather(a, x),
                                      self.config.IMAGES_PER_GPU,
                                      names=["pre_nms_anchors"])

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (z1, y1, x1, z2, y2, x2)]
        boxes = batch_slice([pre_nms_anchors, deltas],
                            lambda x, y: apply_box_deltas(x, y),
                            self.config.IMAGES_PER_GPU,
                            names=["refined_anchors"])

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (z1, y1, x1, z2, y2, x2)]
        window = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
        boxes = batch_slice(boxes, lambda x: clip_boxes(x, window),
                            self.config.IMAGES_PER_GPU,
                            names=["refined_anchors_clipped"])

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non-maximum suppression
        proposals = batch_slice([boxes, scores],
                        lambda x, y: nms_proposals(x, y, self.proposal_count, self.nms_threshold),
                        self.config.IMAGES_PER_GPU,
                        names=["rpn_non_max_suppression"])
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 6)


############################################################
#  ROIAlign Layer
############################################################

def log2(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return tf.log(x) / tf.log(2.0)


class PyramidROIAlign(KE.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.
    Params:
    - pool_shape: [pool_depth, pool_height, pool_width] of the output pooled regions.
                  Usually [7, 7, 7]
    Inputs:
    - boxes: [batch, num_boxes, (z1, y1, x1, z2, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, depth, height, width, channels]
    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_depth, pool_height, pool_width, channels].
    The depth, height, and width are those specific in the pool_shape in the layer constructor.
    """

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (z1, y1, x1, z2, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Image meta
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, depth, height, width, channels]
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        z1, y1, x1, z2, y2, x2 = tf.split(boxes, 6, axis=2)

        h = y2 - y1
        w = x2 - x1

        # Use shape of first image. Images in a batch must have the same size.
        image_shape = mutils.parse_image_meta(image_meta)['image_shape'][0]
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[1] * image_shape[2], tf.float32)
        roi_level = log2(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_depth, pool_height, pool_width, channels]
            cr_roi = cr_3D(feature_maps[i], level_boxes, box_indices,
                        self.pool_shape, method="bilinear") # TODO: 3D crop_and_resize
            pooled.append(cr_roi)

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )


############################################################
#  Detection Target Layer
############################################################

def overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (z1, y1, x1, z2, y2, x2)].
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 6])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_z1, b1_y1, b1_x1, b1_z2, b1_y2, b1_x2 = tf.split(b1, 6, axis=1)
    b2_z2, b2_y1, b2_x1, b2_z2, b2_y2, b2_x2 = tf.split(b2, 6, axis=1)
    z1 = tf.maximum(b1_z1, b2_z2)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    z2 = tf.minimum(b1_z1, b2_z2)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0) * tf.maximum(z2 - z1, 0)
    # 3. Compute unions
    b1_vol = (b1_z2 - b1_z1) * (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_vol = (b2_z2 - b2_z1) * (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_vol + b2_vol - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.
    Inputs:
    proposals: [POST_NMS_ROIS_TRAINING, (z1, y1, x1, z2, y2, x2)] in normalized coordinates.
               Might be zero padded if there are not enough proposals.
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    gt_boxes: [MAX_GT_INSTANCES, (z1, y1, x1, z2, y2, x2)] in normalized coordinates.
    gt_masks: [depth, height, width, MAX_GT_INSTANCES] of boolean type.
    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [TRAIN_ROIS_PER_IMAGE, (z1, y1, x1, z2, y2, x2)] in normalized coordinates
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    deltas: [TRAIN_ROIS_PER_IMAGE, (dz, dy, dx, log(dd), log(dh), log(dw))]
    masks: [TRAIN_ROIS_PER_IMAGE, depth, height, width]. Masks cropped to bbox
           boundaries and resized to neural network output size.
    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = mutils.trim_zeros(proposals, name="trim_proposals")
    gt_boxes, non_zeros = mutils.trim_zeros(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                         name="trim_gt_masks")

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps(proposals, gt_boxes)

    # Compute overlaps with crowd boxes [proposals, crowd_boxes]
    crowd_overlaps = overlaps(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn = lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Assign positive ROIs to GT masks
    # Permute masks to [N, depth, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [3, 0, 1, 2]), -1)
    # Pick the right mask for each ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    # Compute mask targets
    boxes = positive_rois
    if config.USE_MINI_MASK:
        # Transform ROI coordinates from normalized image space
        # to normalized mini-mask space.
        z1, y1, x1, z2, y2, x2 = tf.split(positive_rois, 6, axis=1)
        gt_z1, gt_y1, gt_x1, gt_z2, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 6, axis=1)
        gt_d = gt_z2 - gt_z1
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        z1 = (z1 - gt_z1) / gt_d
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        z2 = (z2 - gt_z1) / gt_d
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([z1, y1, x1, z2, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = cr_3D(tf.cast(roi_masks, tf.float32), boxes,
                                   box_ids, config.MASK_SHAPE)
    # Remove the extra dimension from masks.
    # TODO: print dimensions and verify squeeze is done correctly
    masks = tf.squeeze(masks, axis=4)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)

    ##  TODO: verify padding works for 3D
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0), (0, 0)])

    return rois, roi_gt_class_ids, deltas, masks


class DetectionTargetLayer(KE.Layer):
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.
    Inputs:
    proposals: [batch, N, (z1, y1, x1, z2, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (z1, y1, x1, z2, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, depth, height, width, MAX_GT_INSTANCES] of boolean type
    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (z1, y1, x1, z2, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dd, dy, dx, log(d), log(dh), log(dw)]
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, depth, height, width]
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.
    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # Slice the batch and run a graph for each slice
        names = ["rois", "target_class_ids", "target_deltas", "target_mask"]
        outputs = batch_slice([proposals, gt_class_ids, gt_boxes, gt_masks],
                              lambda w, x, y, z: detection_targets(
                              w, x, y, z, self.config),
                              self.config.IMAGES_PER_GPU, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 6),  # rois
            (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 6),  # deltas
            (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0],
             self.config.MASK_SHAPE[1], self.config.MASK_SHAPE[2])  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None, None]


############################################################
#  Detection Layer
############################################################

def refine_detections(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.
    Inputs:
        rois: [N, (z1, y1, x1, z2, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dz, dy, dx, log(dd), log(dh), log(dw))].
                Class-specific bounding box deltas.
        window: (z1, y1, x1, z2, y2, x2) in normalized coordinates. The part of
            the image that contains the image excluding the padding.
    Returns detections shaped: [num_detections, (z1, y1, x1, z2, y2, x2, class_id, score)]
        where coordinates are normalized.
    """
    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    # Shape: [boxes, (z1, y1, x1, z2, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas(rois, deltas_specific * config.BBOX_STD_DEV)
    # Clip boxes to image window
    refined_rois = clip_boxes(refined_rois, window)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = nms_3D(tf.gather(pre_nms_rois, ixs),
                        tf.gather(pre_nms_scores, ixs),
                        max_output_size=config.DETECTION_MAX_INSTANCES,
                        iou_threshold=config.DETECTION_NMS_THRESHOLD)
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (z1, y1, x1, z2, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections


class DetectionLayer(KE.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.
    Returns:
    [batch, num_detections, (z1, y1, x1, z2, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.
        m = mutils.parse_image_meta(image_meta)
        image_shape = m['image_shape'][0]
        window = norm_boxes(m['window'], image_shape[:3])

        # Run detection refinement graph on each item in the batch
        detections_batch = batch_slice([rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: refine_detections(x, y, w, z, self.config),
            self.config.IMAGES_PER_GPU)

        # Reshape output
        # [batch, num_detections, (z1, y1, x1, z2, y2, x2, class_id, class_score)]
        # in normalized coordinates
        return tf.reshape(detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 8])

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 8)


############################################################
#  Region Proposal Network (RPN)
############################################################

def rpn(feature_map, anchors_per_location, anchor_stride, dim=3):
    """Builds the Region Proposal Network.
    feature_map: backbone features [batch, depth, height, width, pyramid_size]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    Returns:
        rpn_class_logits: [batch, D * H * W * anchors_per_location, 2]
                           Anchor classifier logits (before softmax).
        rpn_probs: [batch, D * H * W * anchors_per_location, 2]
                    Anchor classifier probabilities.
        rpn_bbox: [batch, D * H * W * anchors_per_location, (dz, dy, dx, log(dd), log(dh), log(dw))]
                   Deltas to be applied to anchors.
    """
    # TODO: check if stride of 2 causes alignment issues if the feature map
    # is not even.
    # Shared convolutional base of the RPN
    shared = Conv3D(512, kernel_size=3, padding='same', activation='relu',
                    strides=anchor_stride, name='rpn_conv_shared')(feature_map)

    # Anchor Score. [batch, D, H, W, anchors per location * 2].
    x = Conv3D(anchors_per_location * 2, kernel_size=1, padding='valid',
               activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, D, H, W, anchors per location * (2*dim)]
    # where depth is [x, y, z, log(w), log(h), log(d)]
    x = Conv3D(anchors_per_location * 2 * dim, kernel_size=1, padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, (2*dim)]
    rpn_bbox = Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2 * dim]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def RPN_model(anchor_stride, anchors_per_location, pyramid_size, dim=3):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    anchors_per_location: number of anchors per pixel in the feature map
    pyramid_size: size of the feature pyramid
    dim: dimension of the feature data, i.e. 3 for 3D data

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_class_logits: [batch, D * H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, D * H * W * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, D * H * W * anchors_per_location, (dz, dy, dx, log(dd), log(dh), log(dw))]
              Deltas to be applied to anchors.
    """
    # TODO: verify input_feature_map shape is correct!
    input_feature_map = Input(shape=[None, None, None, pyramid_size],
                              name="input_rpn_feature_map")
    outputs = rpn(input_feature_map, anchors_per_location,
                  anchor_stride, dim)
    return Model([input_feature_map], outputs, name="rpn_model")


############################################################
#  Feature Pyramid Network Heads
############################################################

def fpn_classifier_head(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True,
                         fc_layers_size=1024):
    """Builds the computation graph of the feature pyramid network classifier
    and regressor heads.
    rois: [batch, num_rois, (z1, y1, x1, z2, y2, x2)] Proposal boxes in
          normalized coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers
    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dz, dy, dx, log(dd), log(dh), log(dw))]
                     Deltas to apply to proposal boxes
    """
    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size, pool_size],
                        name="roi_align_classifier")([rois, image_meta] + feature_maps)
    # Two 1024 FC layers (implemented with Conv3D for consistency)
    x = TimeDistributed(Conv3D(fc_layers_size, kernel_size=pool_size, padding="valid"),
                        name="mrcnn_class_conv1")(x)
    x = TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = Activation('relu')(x)
    x = TimeDistributed(Conv3D(fc_layers_size, kernel_size=1),
                        name="mrcnn_class_conv2")(x)
    x = TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = Activation('relu')(x)

    shared = Lambda(lambda x: K.squeeze(K.squeeze(K.squeeze(x, 4), 3), 2),
                       name="pool_squeeze")(x)

    # Classifier head
    mrcnn_class_logits = TimeDistributed(Dense(num_classes),
                                            name='mrcnn_class_logits')(shared)
    mrcnn_probs = TimeDistributed(Activation("softmax"),
                                     name="mrcnn_class")(mrcnn_class_logits)

    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dz, dy, dx, log(dd), log(dh), log(dw))]
    x = TimeDistributed(Dense(num_classes * 6, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dz, dy, dx, log(dd), log(dh), log(dw))]
    s = K.int_shape(x)
    mrcnn_bbox = Reshape((s[1], num_classes, 6), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def fpn_mask_head(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):
    """Builds the computation graph of the mask head of Feature Pyramid Network.
    rois: [batch, num_rois, (z1, y1, x1, z2, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """
    # ROI Pooling
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size, pool_size],
                        name="roi_align_mask")([rois, image_meta] + feature_maps)

    # Conv layers
    x = TimeDistributed(Conv3D(256, kernel_size=3, padding="same"),
                           name="mrcnn_mask_conv1")(x)
    x = TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn1')(x, training=train_bn)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv3D(256, kernel_size=3, padding="same"),
                           name="mrcnn_mask_conv2")(x)
    x = TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn2')(x, training=train_bn)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv3D(256, kernel_size=3, padding="same"),
                           name="mrcnn_mask_conv3")(x)
    x = TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn3')(x, training=train_bn)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv3D(256, kernel_size=3, padding="same"),
                           name="mrcnn_mask_conv4")(x)
    x = TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn4')(x, training=train_bn)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv3DTranspose(256, kernel_size=2, strides=2, activation="relu"),
                           name="mrcnn_mask_deconv")(x)
    x = TimeDistributed(Conv3D(num_classes, kernel_size=1, strides=1, activation="sigmoid"),
                           name="mrcnn_mask")(x)
    return x
