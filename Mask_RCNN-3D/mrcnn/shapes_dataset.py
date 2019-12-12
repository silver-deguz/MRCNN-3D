import os
import sys
import math
import random
import numpy as np
import argparse
import warnings
import pickle
import cv2
import scipy

from collections import OrderedDict
import time
import subprocess
import utils.dataloader_utils as dutils

# batch generator tools from https://github.com/MIC-DKFZ/batchgenerators
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.transforms.spatial_transforms import MirrorTransform as Mirror
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading import SingleThreadedAugmenter
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform
# from batchgenerators.transforms.utility_transforms import ConvertSegToBoundingBoxCoordinates
from batchgenerators.transforms.utility_transforms import NullOperation

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
        # class_ids = np.array([self.class_names.index(s[0]) for s in shapes])

         # SD - change mask type from bool to uint8 and return class_ids as names
        class_ids = np.array([s[0] for s in shapes])
        return mask.astype('uint8'), class_ids #class_ids.astype(np.int32)

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



def get_train_generators(cf, logger, train_dir):
    """
    wrapper function for creating the training batch generator pipeline. returns the train/val generators.
    selects patients according to cv folds (generated by first run/fold of experiment):
    splits the data into n-folds, where 1 split is used for val, 1 split for testing and the rest for training. (inner loop test set)
    If cf.hold_out_test_set is True, adds the test split to the training data.
    """
    ## SD - instantiate Train ShapesDataset
    # Training dataset
    train_dataset = ShapesDataset(num_samples=1000, height=320, width=320)
    train_dataset.load_shapes()
    train_dataset.prepare()

    all_data = load_dataset(cf, logger, train_dataset)
    all_pids_list = np.unique([v['pid'] for (k, v) in all_data.items()])

    train_pids = all_pids_list[:200] # hardcoded these values
    val_pids = all_pids_list[200:300]

    train_data = {k: v for (k, v) in all_data.items() if any(p == v['pid'] for p in train_pids)}
    val_data = {k: v for (k, v) in all_data.items() if any(p == v['pid'] for p in val_pids)}

    logger.info("dataset loaded with: {} train / {} val items".format(len(train_pids), len(val_pids)))
    batch_gen = {}
    print('SD -- update BatchGenerator!!')
    batch_gen['train'] = create_data_gen_pipeline(train_data, cf=cf, do_aug=False)
    batch_gen['val_sampling'] = create_data_gen_pipeline(val_data, cf=cf, do_aug=False)

    # Mode is set to val_sampling in configs, so hits else condition
    if cf.val_mode == 'val_patient':
        batch_gen['val_patient'] = PatientBatchIterator(val_data, cf=cf)
        batch_gen['n_val'] = len(val_pids) if cf.max_val_patients is None else cf.max_val_patients
    else:
        batch_gen['n_val'] = cf.num_val_batches

    return batch_gen


## TODO: Update the way test_data is loaded later
def get_test_generator(cf, logger, test_dir):
    """
    wrapper function for creating the test batch generator pipeline.
    selects patients according to cv folds (generated by first run/fold of experiment)
    If cf.hold_out_test_set is True, gets the data from an external folder instead.
    """
    if cf.hold_out_test_set:
        cf.pp_data_path = cf.pp_test_data_path
        cf.pp_name = cf.pp_test_name
        test_ix = None
    else:
        with open(os.path.join(cf.exp_dir, 'fold_ids.pickle'), 'rb') as handle:
            fold_list = pickle.load(handle)
        _, _, test_ix, _ = fold_list[cf.fold]
        # warnings.warn('WARNING: using validation set for testing!!!')

    ## SD - instantiate Test ShapesDataset
    # Test dataset
    test_dataset = ShapesDataset(test_dir)
    test_dataset.load_shapes(100, height=320, width=320)
    test_dataset.prepare()

    test_data = load_dataset(cf, logger, test_dataset, subset_ixs=test_ix)
    logger.info("data set loaded with: {} test patients from {}".format(len(test_data.keys()), cf.pp_data_path))
    batch_gen = {}
    batch_gen['test'] = PatientBatchIterator(test_data, cf=cf)
    batch_gen['n_test'] = len(test_data.keys())
    return batch_gen


def load_dataset(cf, logger, dataset, subset_ixs=None):
    """
    loads the dataset. if deployed in cloud also copies and unpacks the data to the working directory.
    :param subset_ixs: subset indices to be loaded from the dataset. used e.g. for testing to only load the test folds.
    :return: data: dictionary with one entry per patient (in this case per patient-breast, since they are treated as
    individual images for training) each entry is a dictionary containing respective meta-info as well as paths to the preprocessed
    numpy arrays to be loaded during batch-generation
    """

    # SD - remove pandas dependence and cloud dependence
    # p_df = pd.read_pickle(os.path.join(cf.pp_data_path, cf.input_df_name))
    if subset_ixs is not None:
        # subset_pids = [np.unique(p_df.pid.tolist())[ix] for ix in subset_ixs]
        # p_df = p_df[p_df.pid.isin(subset_pids)]
        # logger.info('subset: selected {} instances from df'.format(len(p_df)))
        subset_pids = [np.unique(dataset.image_ids)[ix] for ix in subset_ixs]
        logger.info('subset: selected {} instances from df'.format(len(subset_pids)))

    # SD - removed since shapes dataset is generated differently
    # class_targets = p_df['class_id'].tolist()
    # pids = p_df.pid.tolist()
    # imgs = [os.path.join(cf.pp_data_path, '{}.npy'.format(pid)) for pid in pids]
    # segs = [os.path.join(cf.pp_data_path,'{}.npy'.format(pid)) for pid in pids]

    pids = list(dataset.image_ids)
    data = OrderedDict()

    # SD - modified based on shapes dataset
    for pid in pids:
        img = dataset.load_image(pid)
        seg, class_ids = dataset.load_mask(pid)
        data[pid] = = {'data': img, 'seg': segs, 'pid': pid, 'class_target': list(class_target)}

    # for ix, pid in enumerate(pids):
        # data[pid] = {'data': imgs[ix], 'seg': segs[ix], 'pid': pid, 'class_target': [class_targets[ix]]}

    # GG Print info
    print("GG load_dataset")
    print(" subset_ixs", subset_ixs)
    # print(" pickle_name", os.path.join(cf.pp_data_path, cf.input_df_name))
    print(" len(pids)", len(pids))
    return data


def create_data_gen_pipeline(data, cf, do_aug=True):
    """
    create mutli-threaded train/val/test batch generation and augmentation pipeline.
    :param data: dictionary containing one dictionary per sample in the train/test subset.
    :param is_training: (optional) whether to perform data augmentation (training) or not (validation/testing)
    :return: multithreaded_generator
    """

    # create instance of batch generator as first element in pipeline.
    data_gen = BatchGenerator(data, batch_size=cf.batch_size, cf=cf)

    # add transformations to pipeline.
    my_transforms = []
    if do_aug:
        mirror_transform = Mirror(axes=np.arange(2, cf.dim+2, 1))
        my_transforms.append(mirror_transform)
        spatial_transform = SpatialTransform(patch_size=cf.patch_size[:cf.dim],
                                             patch_center_dist_from_border=cf.da_kwargs['rand_crop_dist'],
                                             do_elastic_deform=cf.da_kwargs['do_elastic_deform'],
                                             alpha=cf.da_kwargs['alpha'], sigma=cf.da_kwargs['sigma'],
                                             do_rotation=cf.da_kwargs['do_rotation'], angle_x=cf.da_kwargs['angle_x'],
                                             angle_y=cf.da_kwargs['angle_y'], angle_z=cf.da_kwargs['angle_z'],
                                             do_scale=cf.da_kwargs['do_scale'], scale=cf.da_kwargs['scale'],
                                             random_crop=cf.da_kwargs['random_crop'])

        my_transforms.append(spatial_transform)
    else:
        my_transforms.append(CenterCropTransform(crop_size=cf.patch_size[:cf.dim]))

    # GG my_transforms.append(ConvertSegToBoundingBoxCoordinates(cf.dim, get_rois_from_seg_flag=False, class_specific_seg_flag=cf.class_specific_seg_flag))
    my_transforms.append(NullOperation(cf.dim))
    # GG
    my_transforms = []

    all_transforms = Compose(my_transforms)
    # multithreaded_generator = SingleThreadedAugmenter(data_gen, all_transforms)
    multithreaded_generator = MultiThreadedAugmenter(data_gen, all_transforms, num_processes=cf.n_workers, seeds=range(cf.n_workers))
    return multithreaded_generator


############################################################
#  Pytorch Batch Generator
############################################################

class BatchGenerator(SlimDataLoaderBase):
    """
    creates the training/validation batch generator. Samples n_batch_size patients (draws a slice from each patient if 2D)
    from the data set while maintaining foreground-class balance. Returned patches are cropped/padded to pre_crop_size.
    Actual patch_size is obtained after data augmentation.
    :param data: data dictionary as provided by 'load_dataset'.
    :param batch_size: number of patients to sample for the batch
    :return dictionary containing the batch data (b, c, x, y, (z)) / seg (b, 1, x, y, (z)) / pids / class_target
    """
    def __init__(self, data, batch_size, cf):
        super(BatchGenerator, self).__init__(data, batch_size)

        self.cf = cf
# GG
    def generate_train_batch(self):

        batch_data, batch_segs, batch_pids, batch_targets = [], [], [], []
        class_targets_list =  [v['class_target'] for (k, v) in self._data.items()]

        #samples patients towards equilibrium of foreground classes on a roi-level (after randomly sampling the ratio "batch_sample_slack).
        batch_ixs = dutils.get_class_balanced_patients(
            class_targets_list, self.batch_size, self.cf.head_classes - 1, slack_factor=self.cf.batch_sample_slack)
        patients = list(self._data.items())

        for b in batch_ixs:

            patient = patients[b][1]
            all_data = np.load(patient['data'], mmap_mode='r')
            data = all_data[0]
            seg = all_data[1].astype('uint8')
            batch_pids.append(patient['pid'])
            batch_targets.append(patient['class_target'])
            batch_data.append(data[np.newaxis])
            batch_segs.append(seg[np.newaxis])
        # GG
        """
        data = np.array(batch_data)
        seg = np.array(batch_segs).astype(np.uint8)
        class_target = np.array(batch_targets)
        print ("data.shape", data.shape )
        """
        """
        Expected returned values :
        >> mrcnn.train_forward
           img.shape (2, 1, 320, 320)
           gt_class_ids.shape  (2, 1)
           gt_boxes.shape (2, 1, 4)
           batch['roi_masks'].shape  (2, 1, 1, 320, 320)
         rk : 2nd dimension equal to 1, suppose to be the nbr of object in image
         >> mrcnn.train_forward
            img.shape           (bs, c, 320, 320)
            gt_class_ids.shape  (bs, 1)
            gt_boxes.shape      (bs, nobjs, 4)
            batch['roi_masks'].shape  (2, nobjs ???, c ???, 320, 320)
        """
        bSize = len(batch_ixs)
        c = 1
        # Take care could be transposed
        xSize = batch_data[0].shape[1]
        ySize = batch_data[0].shape[2]
        # Nbr of ossurences in the image
        nbr_gt_objects = 3
        data = np.zeros( (bSize, c, xSize, ySize) )
        seg = np.zeros( (bSize, nbr_gt_objects, c, xSize, ySize) )
        bboxes = np.zeros( (bSize, nbr_gt_objects, 4 ) )

        class_target = np.zeros( (bSize, nbr_gt_objects) )
        coord_list = []
        for b in range(len(batch_ixs)):
          data_x = np.array( batch_data[b])
          seg_x = np.array(batch_segs[b]).astype(np.uint8)
          # print ("seg_x.shape ", seg_x.shape )
          seg[b,:,:,:,:] = np.array( [ seg_x, np.flip(seg_x, axis=1), np.flip(seg_x, axis=2) ] )
          # [x_min, y_min, x_max, y_max]
          coord_list = []
          seg_ixs = np.argwhere(seg_x != 0)
          coord_list.append ( [np.min(seg_ixs[:, 1])-1, np.min(seg_ixs[:, 2])-1, np.max(seg_ixs[:, 1])+1,
                                         np.max(seg_ixs[:, 2])+1] )
          seg_1 = np.flip(seg_x, axis=1)
          seg_ixs = np.argwhere( seg_1 != 0 )
          coord_list.append ( [np.min(seg_ixs[:, 1])-1, np.min(seg_ixs[:, 2])-1, np.max(seg_ixs[:, 1])+1,
                                         np.max(seg_ixs[:, 2])+1] )
          seg_2 = np.flip(seg_x, axis=2)
          seg_ixs = np.argwhere( seg_2 != 0 )
          coord_list.append ( [np.min(seg_ixs[:, 1])-1, np.min(seg_ixs[:, 2])-1, np.max(seg_ixs[:, 1])+1,
                                         np.max(seg_ixs[:, 2])+1] )


          # print("coord_list", coord_list )
          data[b,:,:,:] = (data_x + np.flip(data_x, axis=0) + np.flip(data_x, axis=1))/3.0
          bboxes[b, : :] = np.array ( coord_list )
          class_target[b,:] = np.array( [batch_targets[b][0]+1, batch_targets[b][0]+1, batch_targets[b][0]+1 ] )

        return {'data': data, 'seg': seg, 'pid': batch_pids, 'class_target': class_target,
                 'roi_masks': seg, 'roi_labels': class_target, 'bb_target': bboxes }
