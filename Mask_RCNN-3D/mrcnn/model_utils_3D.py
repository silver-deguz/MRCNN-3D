import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import BatchNormalization

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


############################################################
#  Custom Batch Norm Layer
############################################################

class BatchNorm(BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.
    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


############################################################
#  Loss Functions
############################################################

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 6], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def rpn_class_loss(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
    """
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(K.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Cross entropy loss
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_loss(config, target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.
    config: the model config object.
    target_bbox: [batch, max positive anchors, (dz, dy, dx, log(dd), log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dz, dy, dx, log(dd), log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack(target_bbox, batch_counts,
                                   config.IMAGES_PER_GPU)

    loss = smooth_l1_loss(target_bbox, rpn_bbox)

    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def mrcnn_class_loss(target_class_ids, pred_class_logits,
                           active_class_ids):
    """Loss for the classifier head of Mask RCNN.
    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """
    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
    target_class_ids = tf.cast(target_class_ids, 'int64')

    # Find predictions of classes that are not in the dataset.
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss


def mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.
    target_bbox: [batch, num_rois, (dz, dy, dx, log(dd), log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dz, dy, dx, log(dd), log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 6))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 6))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = K.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


def mrcnn_mask_loss(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.
    target_masks: [batch, num_rois, depth, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, depth, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3], mask_shape[4]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4], pred_shape[5]))
    # Permute predicted masks to [N, num_classes, depth, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 4, 1, 2, 3])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = K.switch(tf.size(y_true) > 0,
                    K.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.
    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [D, H, W, C] before resizing or padding.
    image_shape: [D, H, W, C] after resizing and padding
    window: (z1, y1, x1, z2, y2, x2) in pixels. The area of the image where the
            real image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=4
        list(image_shape) +           # size=4
        list(window) +                # size=6 (z1, y1, x1, z2, y2, x2) in image cooredinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta


def parse_image_meta(meta):
    """Parses an array that contains image attributes to its components.
    See compose_image_meta() for more details.
    meta: [batch, meta length] where meta length depends on NUM_CLASSES
    Returns a dict of the parsed values.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:5]
    image_shape = meta[:, 5:9]
    window = meta[:, 9:15]  # (z1, y1, x1, z2, y2, x2) window of image in pixels
    scale = meta[:, 15]
    active_class_ids = meta[:, 16:]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32),
        "active_class_ids": active_class_ids.astype(np.int32),
    }


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.
    meta: [batch, meta length] where meta length depends on NUM_CLASSES
    Returns a dict of the parsed tensors.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:5]
    image_shape = meta[:, 5:9]
    window = meta[:, 9:15]  # (z1, y1, x1, z2, y2, x2) window of image in in pixels
    scale = meta[:, 15]
    active_class_ids = meta[:, 16:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }


def mold_image(images, config):
    """Expects an RGB 3D image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


############################################################
#  Miscellenous Functions
############################################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("",""))
        text += "  {}".format(array.dtype)
    print(text)


def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.
    Returns:
        [N, (depth, height, width)], where N is the number of stages
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [  [int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride)),
            int(math.ceil(image_shape[2] / stride))]
            for stride in config.BACKBONE_STRIDES  ])


# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.
    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]
    return result


def trim_zeros(boxes, name='trim_zeros'):
    """Often boxes are represented with matrices of shape [N, 6] and
    are padded with zeros. This removes zero boxes.
    boxes: [N, 6] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


############################################################
#  Bounding Boxes
############################################################

def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (z1, y1, x1, z2, y2, x2)] in pixel coordinates
    shape: [..., (depth, height, width)] in pixels
    Note: In pixel coordinates (z2, y2, x2) is outside the box. But in
    normalized coordinates it's inside the box.
    Returns:
        [..., (z1, y1, x1, z2, y2, x2)] in normalized coordinates
    """
    d, h, w = tf.split(tf.cast(shape, tf.float32), 3)
    scale = tf.concat([d, h, w, d, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 0., 1., 1., 1.])
    return tf.divide(boxes - shift, scale)


def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [..., (z1, y1, x1, z2, y2, x2)] in normalized coordinates
    shape: [..., (depth, height, width)] in pixels
    Note: In pixel coordinates (z2, y2, x2) is outside the box. But in
    normalized coordinates it's inside the box.
    Returns:
        [..., (z1, y1, x1, z2, y2, x2)] in pixel coordinates
    """
    d, h, w = tf.split(tf.cast(shape, tf.float32), 3)
    scale = tf.concat([d, h, w, d, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 0., 1., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)


# def compute_iou(box, boxes, box_vol, boxes_vol):
#     """Calculates IoU of the given box with the array of the given boxes.
#     box: 1D vector [z1, y1, x1, z2, y2, x2]
#     boxes: [boxes_count, (z1, y1, x1, z2, y2, x2)]
#     box_vol: float. the volume of 'box'
#     boxes_vol: array of length boxes_count.
#
#     Note: the volumes are passed in rather than calculated here for
#     efficiency. Calculate once in the caller to avoid duplicate work.
#     """
#     # Calculate intersection volumes
#     z1 = tf.maximum(box[0], boxes[:, 0])
#     z2 = tf.minimum(box[3], boxes[:, 3])
#     y1 = tf.maximum(box[1], boxes[:, 1])
#     y2 = tf.minimum(box[4], boxes[:, 4])
#     x1 = tf.maximum(box[2], boxes[:, 2])
#     x2 = tf.minimum(box[5], boxes[:, 5])
#     intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0) * tf.maximum(z2 - z1, 0)
#     union = box_vol + boxes_vol[:] - intersection[:]
#     iou = intersection / union
#     return iou

# TODO: verify this works correctly for 3D
# def non_max_suppression(boxes, scores, max_output_size, threshold):
#     """Performs non-maximum suppression and returns indices of kept boxes.
#     boxes: [N, (z1, y1, x1, z2, y2, x2)].
#     scores: 1-D tensor of box scores.
#     threshold: IoU threshold to use for filtering.
#     """
#     assert boxes.shape[0] > 0
#
#     # Compute box volumes
#     z1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     x1 = boxes[:, 2]
#     z2 = boxes[:, 3]
#     y2 = boxes[:, 4]
#     x2 = boxes[:, 5]
#     vol = (z2 - z1) * (y2 - y1) * (x2 - x1)
#
#     # Get indicies of boxes sorted by scores (highest first)
#     ixs = tf.nn.top_k(scores, len(boxes), sorted=True,
#                      name="top_anchors").indices
#     # ixs = tf.argsort(scores)[::-1]
#
#     keep = []
#     while len(ixs) > 0:
#         # Pick top box and add its index to the list
#         i = ixs[0]
#         keep.append(i)
#         # Compute IoU of the picked box with the rest
#         iou = compute_iou(boxes[i], boxes[ixs[1:]], vol[i], vol[ixs[1:]])
#         # Identify boxes with IoU over the threshold. This
#         # returns indices into ixs[1:], so add 1 to get
#         # indices into ixs.
#         remove_ixs = tf.where(iou > threshold)[0] + 1
#         # Remove indices of the picked and overlapped boxes.
#         ixs = tf.gather(ixs, remove_ixs)
#         ixs = tf.gather(ixs, 0)
#     return tf.constant(keep[:max_output_size])

def compute_iou(box, boxes, box_vol, boxes_vol):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [z1, y1, x1, z2, y2, x2]
    boxes: [boxes_count, (z1, y1, x1, z2, y2, x2)]
    box_vol: float. the volume of 'box'
    boxes_vol: array of length boxes_count.

    Note: the volumes are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection volumes
    print('here')
    z1 = np.maximum(box[0], boxes[:, 0])
    z2 = np.minimum(box[3], boxes[:, 3])
    y1 = np.maximum(box[1], boxes[:, 1])
    y2 = np.minimum(box[4], boxes[:, 4])
    x1 = np.maximum(box[2], boxes[:, 2])
    x2 = np.minimum(box[5], boxes[:, 5])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0) * np.maximum(z2 - z1, 0)
    union = box_vol + boxes_vol[:] - intersection[:]
    iou = intersection / union
    return iou

def non_max_suppression(boxes, scores, max_output_size, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (z1, y1, x1, z2, y2, x2)].
    scores: 1-D array of box scores.
    threshold: IoU threshold to use for filtering.

    input are numpy arrays, output is tensor
    """
    assert boxes.shape[0] > 0

    # Compute box volumes
    z1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x1 = boxes[:, 2]
    z2 = boxes[:, 3]
    y2 = boxes[:, 4]
    x2 = boxes[:, 5]
    vol = (z2 - z1) * (y2 - y1) * (x2 - x1)

    # print(scores.numpy())
    # ixs = scores.argsort()[::-1]
    ixs = tf.argsort(scores, direction='DESCENDING')[::-1]
    # ixs = tf.nn.top_k(scores, boxes.shape[0], sorted=True,
                     # name="top_anchors").indices
    pick = []
    while ixs.shape[0] > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], vol[i], vol[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return tf.constant(pick[:max_output_size])
