import os
import random
import datetime
import re
import math
import logging
import h5py
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.engine as KE
from keras.models import Model
from keras.layers import Conv3D, UpSampling3D, MaxPooling3D
from keras.layers import Input, Lambda, Activation, Add, Concatenate

import model_utils_3D as mutils
import utils_3D as utils
from resnet_3D import ResNet50, ResNet101
from model_layers_3D import RPN_model, ProposalLayer

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


############################################################
#  MaskRCNN Class
############################################################

class MaskRCNN3D():
    """Encapsulates the Mask RCNN model functionality.
    The actual Keras model is in the keras_model property.

    This builds on the ResNet + FPN backbone stage.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model, self.rpn_feat_maps = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build Mask R-CNN ResNet+FPN backbone architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        d, h, w = config.IMAGE_SHAPE[:3]
        if d / 2**6 != int(d / 2**6) or h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs image_shape: DxHxWxC
        input_image = Input(shape=[None, None, None, config.IMAGE_SHAPE[3]],
                        name="input_image")
        input_image_meta = Input(shape=[config.IMAGE_META_SIZE],
                            name="input_image_meta")
        if mode == "training":
            # RPN GT
            input_rpn_match = Input(shape=[None, 1],
                            name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = Input(shape=[None, 6],
                            name="input_rpn_bbox", dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = Input(shape=[None],
                                name="input_gt_class_ids", dtype=tf.int32)
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (z1, y1, x1, z2, y2, x2)] in image coordinates
            input_gt_boxes = Input(shape=[None, 6],
                                name="input_gt_boxes", dtype=tf.float32)
            # Normalize coordinates
            gt_boxes = Lambda(lambda x: mutils.norm_boxes(
                x, K.shape(input_image)[1:4]))(input_gt_boxes) ## TODO: verify shape is correct for 3D
            # 3. GT Masks (zero padded)
            # [batch, depth, height, width, MAX_GT_INSTANCES]
            # if config.USE_MINI_MASK:
            #     input_gt_masks = Input(
            #         shape=[config.MINI_MASK_SHAPE[0],
            #                config.MINI_MASK_SHAPE[1],
            #                config.MINI_MASK_SHAPE[2], None],
            #         name="input_gt_masks", dtype=bool)
            # else:
            #     input_gt_masks = Input(
            #         shape=[config.IMAGE_SHAPE[0],
            #                config.IMAGE_SHAPE[1],
            #                config.IMAGE_SHAPE[2], None],
            #         name="input_gt_masks", dtype=bool)
        elif mode == "inference":
            # Anchors in normalized coordinates
            input_anchors = Input(shape=[None, 6], name="input_anchors")

        # Build the shared convolutional layers.
        # -- Bottom-up Layers -- #
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the head (stage 5), so we pick the 4th item in the list.
        if callable(config.BACKBONE):
            _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True,
                                                train_bn=config.TRAIN_BN)
        else: # Can also use ResNet50 function
            _, C2, C3, C4, C5 = ResNet50(input_image, stage5=True,
                                          train_bn=config.TRAIN_BN)

        # -- Top-down Layers -- #
        # TODO: add assert to varify feature map sizes match what's in config
        P5 = Conv3D(config.TOP_DOWN_PYRAMID_SIZE, kernel_size=1, name='fpn_c5p5')(C5)
        P4 = Add(name="fpn_p4add")([
            UpSampling3D(size=2, name="fpn_p5upsampled")(P5),
            Conv3D(config.TOP_DOWN_PYRAMID_SIZE, kernel_size=1, name='fpn_c4p4')(C4)])
        P3 = Add(name="fpn_p3add")([
            UpSampling3D(size=2, name="fpn_p4upsampled")(P4),
            Conv3D(config.TOP_DOWN_PYRAMID_SIZE, kernel_size=1, name='fpn_c3p3')(C3)])
        P2 = Add(name="fpn_p2add")([
            UpSampling3D(size=2, name="fpn_p3upsampled")(P3),
            Conv3D(config.TOP_DOWN_PYRAMID_SIZE, kernel_size=1, name='fpn_c2p2')(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = Conv3D(config.TOP_DOWN_PYRAMID_SIZE, kernel_size=3, padding="SAME", name="fpn_p2")(P2)
        P3 = Conv3D(config.TOP_DOWN_PYRAMID_SIZE, kernel_size=3, padding="SAME", name="fpn_p3")(P3)
        P4 = Conv3D(config.TOP_DOWN_PYRAMID_SIZE, kernel_size=3, padding="SAME", name="fpn_p4")(P4)
        P5 = Conv3D(config.TOP_DOWN_PYRAMID_SIZE, kernel_size=3, padding="SAME", name="fpn_p5")(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = MaxPooling3D(pool_size=3, strides=2, name="fpn_p6")(P5)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            # A hack to get around Keras's bad support for constants
            anchors = Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
        else:
            anchors = input_anchors

        # RPN Model
        rpn = RPN_model(config.RPN_ANCHOR_STRIDE,
                              len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
            else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config)([rpn_class, rpn_bbox, anchors])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            # active_class_ids = Lambda(
                # lambda x: mutils.parse_image_meta(x)["active_class_ids"]
                # )(input_image_meta)

             # RPN Losses
            rpn_class_loss = Lambda(lambda x: mutils.rpn_class_loss(*x),
                name="rpn_class_loss")([input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = Lambda(lambda x: mutils.rpn_bbox_loss(config, *x),
                name="rpn_bbox_loss")([input_rpn_bbox, input_rpn_match, rpn_bbox])

            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       rpn_class_loss, rpn_bbox_loss]
        else:
            outputs = [rpn_class_logits, rpn_class, rpn_bbox]

        inputs = [input_image, input_image_meta,
                  input_rpn_match, input_rpn_bbox,
                  input_gt_class_ids, input_gt_boxes]

        model = Model(inputs=inputs, outputs=outputs,
                             name='resnet_fpn_backbone')
        return model, rpn_feature_maps

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.
        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir))
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        # Conditional import to support versions of Keras before 2.2
        # TODO: remove in about 6 months (end of 2018)
        try:
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        # backbone_shapes = mutils.compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                self.config.BACKBONE_SHAPES,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:3])
        return self._anchor_cache[tuple(image_shape)]

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, clipnorm=self.config.GRADIENT_CLIP_NORM)

        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["rpn_class_loss", "rpn_bbox_loss"]
        # loss_names = ["rpn_class_loss",  "rpn_bbox_loss",
        #               "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]

        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile, pass arguments to Keras compile function
        self.keras_model.compile(optimizer=optimizer,
                                 loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)
