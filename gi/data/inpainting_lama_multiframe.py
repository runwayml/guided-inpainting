import glob
import logging
import os
import random
import fnmatch

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image as Image
from omegaconf import open_dict, OmegaConf
from skimage.feature import canny
from skimage.transform import rescale, resize
from torch.utils.data import Dataset, IterableDataset, DataLoader, DistributedSampler, ConcatDataset

from gi.modules.lama.saicinpainting.training.data.aug import IAAAffine2, IAAPerspective2
from gi.modules.lama.saicinpainting.training.data.masks import get_mask_generator

LOGGER = logging.getLogger(__name__)


class InpaintingTrainDataset(Dataset):
    def __init__(self, paths, n_references, mask_generator, transform_shared,
                 transform_individual):
        self.in_files = paths
        self.n_references = n_references
        self.mask_generator = mask_generator
        self.transform_shared = transform_shared
        self.transform_individual = transform_individual
        self.iter_i = 0

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data = self.transform_shared(image=img)
        frame = data['image']
        frame = self.transform_individual(image=frame)["image"]
        images = [frame]
        for i in range(self.n_references):
            frame = A.ReplayCompose.replay(data['replay'], image=img)["image"]
            frame = self.transform_individual(image=frame)["image"]
            images.append(frame)
        images = np.stack(images, 0) # thwc

        img_t = np.transpose(images[0], (2, 0, 1))
        # TODO: maybe generate mask before augmentations? slower, but better for segmentation-based masks
        masks = list()
        for _ in range(self.n_references+1):
            mask = self.mask_generator(img_t, iter_i=self.iter_i)
            mask = mask.transpose(1,2,0)
            self.iter_i += 1
            masks.append(mask)
        masks = np.stack(masks, 0)

        # NOTE: used to be chw in [0,1] but we return hwc in [-1,1] for
        # compatibility with other models
        result = dict(images=images,
                      masks=masks,
                      masked_images=images*(1-masks))
        for k in ["images", "masks", "masked_images"]:
            result[k] = (result[k]*2.0-1.0)
        return result


def get_transforms(transform_variant, out_size, easy=False):
    assert transform_variant == "distortions"
    if transform_variant == 'default':
        transform = A.Compose([
            A.RandomScale(scale_limit=0.2),  # +/- 20%
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions':
        if easy:
            max_shift = 25
            transform_shared = A.ReplayCompose([
                A.PadIfNeeded(min_height=out_size+max_shift,
                              min_width=out_size+max_shift),
                A.RandomCrop(height=out_size+max_shift, width=out_size+max_shift),
                A.HorizontalFlip(),
                A.CLAHE(),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            ])
            transform_individual = A.Compose([
                A.PadIfNeeded(min_height=out_size, min_width=out_size),
                A.RandomCrop(height=out_size, width=out_size),
                A.ToFloat()
            ])
        else:
            max_shift = 200
            transform_shared = A.ReplayCompose([
                A.PadIfNeeded(min_height=out_size+max_shift,
                              min_width=out_size+max_shift),
                A.RandomCrop(height=out_size+max_shift, width=out_size+max_shift),
                A.HorizontalFlip(),
                A.CLAHE(),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            ])
            transform_individual = A.Compose([
                IAAPerspective2(scale=(0.0, 0.06)),
                IAAAffine2(scale=(0.7, 1.3),
                           rotate=(-40, 40),
                           shear=(-0.1, 0.1)),
                A.PadIfNeeded(min_height=out_size, min_width=out_size),
                A.OpticalDistortion(),
                A.RandomCrop(height=out_size, width=out_size),
                A.RandomBrightnessContrast(brightness_limit=0.01, contrast_limit=0.01),
                A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=3, val_shift_limit=1),
                A.ToFloat()
            ])
    elif transform_variant == 'distortions_scale05_1':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.5, 1.0),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale03_12':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.3, 1.2),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale03_07':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.3, 0.7),  # scale 512 to 256 in average
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_light':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.02)),
            IAAAffine2(scale=(0.8, 1.8),
                       rotate=(-20, 20),
                       shear=(-0.03, 0.03)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'non_space_transform':
        transform = A.Compose([
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'no_augs':
        transform = A.Compose([
            A.ToFloat()
        ])
    else:
        raise ValueError(f'Unexpected transform_variant {transform_variant}')
    return transform_shared, transform_individual


def make_default_train_dataset(root, filelist, kind='default', out_size=512,
                               mask_gen_kwargs=None,
                               transform_variant='default',
                               mask_generator_kind="mixed",
                               easy=False, **kwargs):
    if kind != 'default':
        raise ValueError(f"Dropped support for other datasets: {kind}")

    LOGGER.info(f'Make train dataloader from {filelist}. Using mask generator={mask_generator_kind}')

    with open(filelist, "r") as f:
        paths = f.read().splitlines()
    paths = [os.path.join(root, x) for x in paths]

    mask_generator = get_mask_generator(kind=mask_generator_kind, kwargs=mask_gen_kwargs)
    transform_shared, transform_individual = get_transforms(transform_variant,
                                                            out_size, easy=easy)

    dataset = InpaintingTrainDataset(paths=paths,
                                     mask_generator=mask_generator,
                                     transform_shared=transform_shared,
                                     transform_individual=transform_individual,
                                     **kwargs)
    return dataset


def make_default_val_dataset(indir, kind='default', out_size=512, **kwargs):
    if kind != 'default':
        raise ValueError(f"Dropped support for other datasets: {kind}")

    if OmegaConf.is_list(indir) or isinstance(indir, (tuple, list)):
        return ConcatDataset([
            make_default_val_dataset(idir, kind=kind, out_size=out_size,
                                     **kwargs) for idir in indir 
        ])

    LOGGER.info(f'Make val dataloader {kind} from {indir}')
    mask_generator = get_mask_generator(kind=kwargs.get("mask_generator_kind"),
                                        kwargs=kwargs.get("mask_gen_kwargs"))

    dataset = InpaintingEvaluationDataset(indir, **kwargs)
    return dataset



###### validation


def load_image(fname, mode='RGB', return_orig=False):
    img = np.array(Image.open(fname).convert(mode))
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')


def pad_tensor_to_modulo(img, mod):
    batch_size, channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return F.pad(img, pad=(0, out_width - width, 0, out_height - height), mode='reflect')


def scale_image(img, factor, interpolation=cv2.INTER_AREA):
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = np.transpose(img, (1, 2, 0))

    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)

    if img.ndim == 2:
        img = img[None, ...]
    else:
        img = np.transpose(img, (2, 0, 1))
    return img


class InpaintingEvaluationDataset(Dataset):
    def __init__(self, datadir, img_suffix='.jpg', pad_out_to_modulo=None, scale_factor=None):
        self.datadir = datadir
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, '**', '*mask*.png'), recursive=True)))
        self.img_filenames = [fname.rsplit('_mask', 1)[0] + img_suffix for fname in self.mask_filenames]
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, i):
        image = load_image(self.img_filenames[i], mode='RGB')
        mask = load_image(self.mask_filenames[i], mode='L')
        result = dict(image=image, mask=mask[None, ...])

        if self.scale_factor is not None:
            result['image'] = scale_image(result['image'], self.scale_factor)
            result['mask'] = scale_image(result['mask'], self.scale_factor, interpolation=cv2.INTER_NEAREST)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)

        # NOTE: used to be chw in [0,1] but we return hwc in [-1,1] for
        # compatibility with other models
        for k in ["image", "mask"]:
            result[k] = (result[k]*2.0-1.0).transpose(1,2,0)

        return result


###### propagation

default_mask_config = {
    "mask_gen_kwargs": {
        "irregular_proba": 1,
        "irregular_kwargs": {
            "max_angle": 4,
            "max_len": 200,
            "max_width": 100,
            "max_times": 5,
            "min_times": 1,
        },

        "box_proba": 1,
        "box_kwargs": {
            "margin": 10,
            "bbox_min_size": 30,
            "bbox_max_size": 150,
            "max_times": 4,
            "min_times": 1,
        },

        "segm_proba": 0,
    },

    "transform_variant": "distortions",
}


class PRNGMixin(object):
    """Adds a prng property which is a numpy RandomState which gets
    reinitialized whenever the pid changes to avoid synchronized sampling
    behavior when used in conjunction with multiprocessing."""

    @property
    def prng(self):
        currentpid = os.getpid()
        if getattr(self, "_initpid", None) != currentpid:
            self._initpid = currentpid
            self._prng = np.random.RandomState()
        return self._prng


class LamaPropagation(Dataset, PRNGMixin):
    def __init__(self, **kwargs):
        self.clean_prob = kwargs.pop("clean_prob",
                                     1.0/kwargs["n_references"])
        for k in default_mask_config:
            if not k in kwargs:
                kwargs[k] = default_mask_config[k]
        self.base_data = make_default_train_dataset(**kwargs)

    def __len__(self):
        return len(self.base_data)

    def __getitem__(self, i):
        example = self.base_data[i]
        T = example["images"].shape[0]
        values = list()
        for i in range(T):
            if self.prng.random() < self.clean_prob:
                values.append(
                    np.concatenate([example["images"][i],
                                    -1*np.ones_like(example["masks"][i])],
                                   axis=-1)
                )
            else:
                values.append(
                    np.concatenate([example["masked_images"][i],
                                    example["masks"][i]],
                                   axis=-1)
                )

        return {
            "rgbs": np.concatenate([example["images"],
                                    example["masks"]],
                                   axis=-1),
            "keys": np.concatenate([example["masked_images"],
                                    example["masks"]],
                                   axis=-1),
            "values": np.stack(values, axis=0),
            "targets": example["images"],
        }



class LamaGI(Dataset, PRNGMixin):
    def __init__(self, **kwargs):
        self.clean_prob = kwargs.pop("clean_prob",
                                     1.0/kwargs["n_references"])
        for k in default_mask_config:
            if not k in kwargs:
                kwargs[k] = default_mask_config[k]
        self.base_data = make_default_train_dataset(**kwargs)

    def __len__(self):
        return len(self.base_data)

    def __getitem__(self, i):
        example = self.base_data[i]
        T = example["images"].shape[0]

        image = example["images"][0]
        mask = example["masks"][0]
        srcs = example["images"][1:]
        srcs_masks = example["masks"][1:]
        return {
            "image": image,
            "mask": mask,
            "srcs": srcs,
            "srcs_masks": srcs_masks,
        }


class LamaGIValidation(Dataset):
    def __init__(self, filenames, n_references, pad_out_to_modulo=None, scale_factor=None):
        self.n_references = n_references

        # image_XXX_mask.png
        # image_XXX.png
        # image_XXX_src_YYY_m.png
        # image_XXX_src_YYY.png
        #filenames = sorted(list(glob.glob(os.path.join(self.datadir, '*.png'))))
        with open(filenames, "r") as f:
            filenames = f.read().splitlines()
        self.mask_filenames = [fname for fname in filenames
                               if fnmatch.fnmatch(fname, "*image*mask*.png")]
        self.img_filenames = [fname.rsplit('_mask', 1)[0] + ".png" for
                              fname in self.mask_filenames]
        self.srcs_masks_filenames_by_img = dict()
        self.srcs_filenames_by_img = dict()
        for parent in self.img_filenames:
            self.srcs_masks_filenames_by_img[parent] = list()
            self.srcs_filenames_by_img[parent] = list()
        for fname in filenames:
            if fnmatch.fnmatch(fname, "*image*_src*_m.png"):
                parent = fname.rsplit("_src", 1)[0]+".png"
                self.srcs_masks_filenames_by_img[parent].append(fname)
                src = fname.rsplit("_m.png", 1)[0]+".png"
                self.srcs_filenames_by_img[parent].append(src)

        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, i):
        image = load_image(self.img_filenames[i], mode='RGB')
        mask = load_image(self.mask_filenames[i], mode='L')
        mask[mask<0.5] = 0
        mask[mask>=0.5] = 1
        mask = mask[None]

        srcs = list()
        srcs_masks = list()
        for t in range(self.n_references):
            srcs.append(load_image(
                self.srcs_filenames_by_img[self.img_filenames[i]][t],
                mode='RGB'))
            src_mask = load_image(
                self.srcs_masks_filenames_by_img[self.img_filenames[i]][t],
                mode='L')
            src_mask[src_mask<0.5] = 0
            src_mask[src_mask>=0.5] = 1
            src_mask = src_mask[None]
            srcs_masks.append(src_mask)
        srcs = np.stack(srcs)
        srcs_masks = np.stack(srcs_masks)

        result = dict(image=image, mask=mask, srcs=srcs, srcs_masks=srcs_masks)

        if self.scale_factor is not None:
            result['image'] = scale_image(result['image'], self.scale_factor)
            result['mask'] = scale_image(result['mask'], self.scale_factor, interpolation=cv2.INTER_NEAREST)
            for t in range(srcs_masks.shape[0]):
                result["srcs"][t] = scale_image(result["srcs"][t],
                                                self.scale_factor)
                result["srcs_masks"][t] = scale_image(result["srcs_masks"][t],
                                                      self.scale_factor,
                                                      interpolation=cv2.INTER_NEAREST)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)
            for t in range(srcs_masks.shape[0]):
                result["srcs"][t] = pad_img_to_modulo(result["srcs"][t],
                                                      self.pad_out_to_modulo)
                result["srcs_masks"][t] = pad_img_to_modulo(result["srcs_masks"][t],
                                                            self.pad_out_to_modulo)

        # NOTE: used to be chw in [0,1] but we return hwc in [-1,1] for
        # compatibility with other models
        for k in ["image", "mask"]:
            result[k] = (result[k]*2.0-1.0).transpose(1,2,0)
        # tchw in 01 to thwc -11
        for k in ["srcs", "srcs_masks"]:
            result[k] = (result[k]*2.0-1.0).transpose(0,2,3,1)

        return result
