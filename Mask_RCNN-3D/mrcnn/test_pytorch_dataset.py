import os
import sys
import math
import random
import numpy as np
import argparse
import time
import pickle
import cv2
import torch


sys.path.append(os.path.dirname(os.path.realpath('__file__')))
root_dir = os.path.dirname(os.path.realpath('__file__'))
print(root_dir)

exp_name = 'shapes'
image_height = image_width = 320
train_dir = os.path.join(root_dir, exp_name, 'train')
val_dir = os.path.join(root_dir, exp_name, 'val')
print(train_dir)
print(val_dir)

############################################################
#  Dataset
############################################################

class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def __init__(self, out_dir):
        super(ShapesDataset, self).__init__()
        self.out_dir = out_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def load_shapes(self, count, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "square")
        self.add_class("shapes", 2, "circle")
        self.add_class("shapes", 3, "triangle")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            bg_color, shapes = self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, shapes=shapes)

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape, color, dims in info['shapes']:
            image = self.draw_shape(image, shape, dims, color)
        return image

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
                                                shape, dims, 1)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def draw_shape(self, image, shape, dims, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x, y, s = dims
        if shape == 'square':
            cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
        elif shape == "circle":
            cv2.circle(image, (x, y), s, color, -1)
        elif shape == "triangle":
            points = np.array([[(x, y-s),
                                (x-s/math.sin(math.radians(60)), y+s),
                                (x+s/math.sin(math.radians(60)), y+s),
                                ]], dtype=np.int32)
            cv2.fillPoly(image, points, color)
        return image

    def random_shape(self, height, width):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        """
        # Shape
        shape = random.choice(["square", "circle", "triangle"])
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height//4)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        """
        # Pick random background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # Generate a few random shapes and record their
        # bounding boxes
        shapes = []
        boxes = []
        N = random.randint(1, 4)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y-s, x-s, y+s, x+s])
        # Apply non-max suppression wit 0.3 threshold to avoid
        # shapes covering each other
        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.3)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes

    def save_image_and_mask(self, image_id):
        img = self.load_image(image_id)
        seg, class_id = self.load_mask(image_id)
        out = np.concatenate((img, seg), axis=2)
        out_path = os.path.join(self.out_dir, '{}.npy'.format(image_id))
        self.image_info[image_id]['path'] = out_path
        np.save(out_path, out)

        with open(os.path.join(self.out_dir, 'meta_info_{}.pickle'.format(image_id)), 'wb') as handle:
            pickle.dump([out_path, class_id, str(image_id)], handle)


############################################################
#  Utils from Matterport version
############################################################

def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.
    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.
    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)),
                       preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop


def resize_mask(mask, scale, padding, crop=None):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.
    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to reduce memory load.
    Mini-masks can be resized back to image scale using expand_masks()
    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
        m = mask[:, :, i].astype(bool)
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = resize(m, mini_shape)
        mini_mask[:, :, i] = np.around(m).astype(np.bool)
    return mini_mask


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)



############################################################
#  Execute Training
############################################################

def load_image_gt(dataset, config, image_id, augment=False, use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: If true, apply random image augmentation. Currently, only horizontal
        flipping is offered.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    image, window, scale, padding, crop = resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = resize_mask(mask, scale, padding, crop)

    # Random horizontal flips.
    if augment:
        logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]

    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = extract_bboxes(mask)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # Image meta data
#     image_meta = compose_image_meta(image_id, original_shape, image.shape,
#                                     window, scale, active_class_ids)
    return image, class_ids, bbox, mask


def train(logger, dataset_train, net):
    """
    perform the training routine for a given fold. saves plots and selected parameters to the experiment dir
    specified in the configs.
    """
    logger.info('performing training in {}D over fold {} on experiment {} with model {}'.format(
        cf.dim, cf.fold, cf.exp_dir, cf.model))

    # net = net(cf, logger).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=cf.learning_rate[0], weight_decay=cf.weight_decay)
    model_selector = utils.ModelSelector(cf, logger)
    train_evaluator = Evaluator(cf, logger, mode='train')
    val_evaluator = Evaluator(cf, logger, mode=cf.val_mode)

    starting_epoch = 1

    # prepare monitoring
    monitor_metrics, TrainingPlot = utils.prepare_monitoring(cf)

    if cf.resume_to_checkpoint:
        starting_epoch, monitor_metrics = utils.load_checkpoint(cf.resume_to_checkpoint, net, optimizer)
        logger.info('resumed to checkpoint {} at epoch {}'.format(cf.resume_to_checkpoint, starting_epoch))

    logger.info('loading dataset and initializing batch generators...')
#     batch_gen = data_loader.get_train_generators(cf, logger)


    for epoch in range(starting_epoch, cf.num_epochs + 1):
        logger.info('starting training epoch {}'.format(epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = cf.learning_rate[epoch - 1]

        start_time = time.time()

        net.train() # function built into nn.module
        train_results_list = []
        num_train_batches = len(dataset_train.image_ids)
        for bix in range(num_train_batches):
#             batch = next(batch_gen['train'])
            batch = {}
            image, class_id, bbox, mask = load_image_gt(dataset_train, config, bix)

            batch['data'] = image
            batch['roi_labels'] = class_id
            batch['bb_target'] = bbox
            batch['roi_masks'] = mask
            batch['pid'] = dataset_train.image_ids[0]

            tic_fw = time.time()
            results_dict = net.train_forward(batch) ### TODO: ensure this works with mods
            tic_bw = time.time()
            optimizer.zero_grad()
            results_dict['torch_loss'].backward()
            optimizer.step()
            logger.info('tr. batch {0}/{1} (ep. {2}) fw {3:.3f}s / bw {4:.3f}s / total {5:.3f}s || '
                        .format(bix + 1, num_train_batches, epoch, tic_bw - tic_fw,
                                time.time() - tic_bw, time.time() - tic_fw) + results_dict['logger_string'])
            train_results_list.append([results_dict['boxes'], batch['pid']])
            monitor_metrics['train']['monitor_values'][epoch].append(results_dict['monitor_values'])

        _, monitor_metrics['train'] = train_evaluator.evaluate_predictions(train_results_list, monitor_metrics['train'])
        train_time = time.time() - start_time

        logger.info('starting validation in mode {}.'.format(cf.val_mode))
        with torch.no_grad():
            net.eval()
        #     if cf.do_validation:
        #         val_results_list = []
        #         val_predictor = Predictor(cf, net, logger, mode='val')
        #         num_val_batches = len(dataset_val.image_ids)
        #         for _ in range(num_val_batches):
        #             # batch = next(batch_gen[cf.val_mode])
        #
        #             if cf.val_mode == 'val_patient':
        #                 results_dict = val_predictor.predict_patient(batch)
        #             elif cf.val_mode == 'val_sampling':
        #                 results_dict = net.train_forward(batch, is_validation=True)
        #             val_results_list.append([results_dict['boxes'], batch['pid']])
        #             monitor_metrics['val']['monitor_values'][epoch].append(results_dict['monitor_values'])
        #
        #         _, monitor_metrics['val'] = val_evaluator.evaluate_predictions(val_results_list, monitor_metrics['val'])
        #         model_selector.run_model_selection(net, optimizer, monitor_metrics, epoch)

            # update monitoring and prediction plots
            TrainingPlot.update_and_save(monitor_metrics, epoch)
            epoch_time = time.time() - start_time
            logger.info('trained epoch {}: took {} sec. ({} train / {} val)'.format(
                epoch, epoch_time, train_time, epoch_time-train_time))
            # batch = next(batch_gen['val_sampling'])
            # results_dict = net.train_forward(batch, is_validation=True)
            # logger.info('plotting predictions from validation sampling.')
            # plot_batch_prediction(batch, results_dict, cf)

        print('End of epoch {}'.format(starting_epoch))

# def test(logger):
#     """
#     perform testing for a given fold (or hold out set). save stats in evaluator.
#     """
#     logger.info('starting testing model of fold {} in exp {}'.format(cf.fold, cf.exp_dir))
#     net = model.net(cf, logger).cuda()
#     test_predictor = Predictor(cf, net, logger, mode='test')
#     test_evaluator = Evaluator(cf, logger, mode='test')
#     batch_gen = data_loader.get_test_generator(cf, logger)
#     test_results_list = test_predictor.predict_test_set(batch_gen, return_results=True)
#     test_evaluator.evaluate_predictions(test_results_list)
#     test_evaluator.score_test_df()



if __name__ == "__main__":
    # Training dataset
    dataset_train = ShapesDataset(train_dir)
    dataset_train.load_shapes(500, image_height, image_width)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ShapesDataset(val_dir)
    dataset_val.load_shapes(50, image_height, image_width)
    dataset_val.prepare()

    # Testing dataset creation
    image_ids = np.random.choice(dataset_train.image_ids, 4)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        print(image.shape)
        print(mask.shape)
        print(dataset_train.image_info[image_id],'\n')

    import utils.exp_utils as utils
    from evaluator import Evaluator
    from predictor import Predictor
    from mrcnn import net

    # Begin training
    # cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, args.use_stored_settings)
    # cf.slurm_job_id = args.slurm_job_id

    # logger = utils.get_logger(exp_dir)
    # model = net(cf, logger).cuda()
    # train(logger, dataset_train, model)
