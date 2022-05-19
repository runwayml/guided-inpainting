import copy, os
import logging
from typing import Dict, Tuple

from omegaconf import OmegaConf
from einops import rearrange

import pandas as pd
import pytorch_lightning as ptl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image
from pytorch_lightning.utilities.apply_func import apply_to_collection

from gi.modules.lama.saicinpainting.evaluation import make_evaluator
from gi.modules.lama.saicinpainting.training.data.datasets import make_default_train_dataloader, make_default_val_dataloader
from gi.modules.lama.saicinpainting.training.losses.adversarial import make_discrim_loss
from gi.modules.lama.saicinpainting.training.losses.perceptual import PerceptualLoss, ResNetPL
from gi.modules.lama.saicinpainting.training.modules import make_generator, make_discriminator
from gi.modules.lama.saicinpainting.training.visualizers import make_visualizer
from gi.modules.lama.saicinpainting.utils import add_prefix_to_keys, average_dicts, set_requires_grad, flatten_dict, get_has_ddp_rank, get_ramp
from gi.modules.lama.saicinpainting.training.data.datasets import make_constant_area_crop_params
from gi.modules.lama.saicinpainting.training.losses.distance_weighting import make_mask_distance_weighter
from gi.modules.lama.saicinpainting.training.losses.feature_matching import feature_matching_loss, masked_l1_loss
from gi.modules.lama.saicinpainting.training.modules.fake_fakes import FakeFakesGenerator
from gi.modules.lama.saicinpainting.training.visualizers.base import visualize_mask_and_images_batch
from gi.modules.RAFT.flow_utils import resized_flow_sample

from gi.main import instantiate_from_config
from torch.optim.lr_scheduler import LambdaLR

LOGGER = logging.getLogger(__name__)


def make_optimizer(parameters, kind='adamw', **kwargs):
    if kind == 'adam':
        optimizer_class = torch.optim.Adam
    elif kind == 'adamw':
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(f'Unknown optimizer kind {kind}')
    return optimizer_class(parameters, **kwargs)


def update_running_average(result: nn.Module, new_iterate_model: nn.Module, decay=0.999):
    with torch.no_grad():
        res_params = dict(result.named_parameters())
        new_params = dict(new_iterate_model.named_parameters())

        for k in res_params.keys():
            res_params[k].data.mul_(decay).add_(new_params[k].data, alpha=1 - decay)


def make_multiscale_noise(base_tensor, scales=6, scale_mode='bilinear'):
    batch_size, _, height, width = base_tensor.shape
    cur_height, cur_width = height, width
    result = []
    align_corners = False if scale_mode in ('bilinear', 'bicubic') else None
    for _ in range(scales):
        cur_sample = torch.randn(batch_size, 1, cur_height, cur_width, device=base_tensor.device)
        cur_sample_scaled = F.interpolate(cur_sample, size=(height, width), mode=scale_mode, align_corners=align_corners)
        result.append(cur_sample_scaled)
        cur_height //= 2
        cur_width //= 2
    return torch.cat(result, dim=1)


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class BaseInpaintingTrainingModule(ptl.LightningModule):
    def __init__(self, config, use_ddp, *args,  predict_only=False, visualize_each_iters=100,
                 average_generator=False, generator_avg_beta=0.999, average_generator_start_step=30000,
                 average_generator_period=10, store_discr_outputs_for_vis=False,
                 ckpt_path=None, ignore_keys=[], scheduler_config=None,
                 visualize_attn=False, validation_keys=["default"],
                 monitor=None, flow_cfg=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        LOGGER.info('BaseInpaintingTrainingModule init called')
        if monitor is not None:
            self._monitor = monitor
            self.monitor = "ckpt_monitor"
            print(f"Monitoring {monitor} as {self.monitor}.")

        self.config = config

        self.generator = make_generator(config, **self.config.generator)
        self.use_ddp = use_ddp

        if flow_cfg is not None:
            print("Instantiating flow module.")
            self.flow = instantiate_from_config(flow_cfg)
            self.flow.eval()
            self.flow.train = disabled_train
        else:
            print("Using no flow module.")
            self.flow = None

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.visualize_attn = visualize_attn
        self.validation_keys = validation_keys

        if not get_has_ddp_rank():
            LOGGER.info(f'Generator\n{self.generator}')

        if not predict_only:
            self.save_hyperparameters(self.config)
            self.discriminator = make_discriminator(**self.config.discriminator)
            self.adversarial_loss = make_discrim_loss(**self.config.losses.adversarial)

            for val_key in self.validation_keys:
                setattr(self, "val_evaluator_raw_{}".format(val_key), 
                        make_evaluator(kind="default",
                                       inpainted_key="predicted_image",
                                       integral_kind="lpips_fid100_f1"))
                setattr(self, "val_evaluator_stitched_{}".format(val_key),
                        make_evaluator(kind="default",
                                       inpainted_key="inpainted",
                                       integral_kind="lpips_fid100_f1"))

            if not get_has_ddp_rank():
                LOGGER.info(f'Discriminator\n{self.discriminator}')

            extra_val = self.config.get('data', dict())
            extra_val = extra_val.get('extra_val', ())
            if extra_val:
                self.extra_val_titles = list(extra_val)
                self.extra_evaluators = nn.ModuleDict({k: make_evaluator(**self.config.evaluator)
                                                       for k in extra_val})
            else:
                self.extra_evaluators = {}

            self.average_generator = average_generator
            self.generator_avg_beta = generator_avg_beta
            self.average_generator_start_step = average_generator_start_step
            self.average_generator_period = average_generator_period
            self.generator_average = None
            self.last_generator_averaging_step = -1
            self.store_discr_outputs_for_vis = store_discr_outputs_for_vis

            # default: 10
            if self.config.losses.get("l1", {"weight_known": 0})['weight_known'] > 0:
                self.loss_l1 = nn.L1Loss(reduction='none')

            # default: no mse
            if self.config.losses.get("mse", {"weight": 0})['weight'] > 0:
                self.loss_mse = nn.MSELoss(reduction='none')
            
            # default: 0
            if self.config.losses.perceptual.weight > 0:
                self.loss_pl = PerceptualLoss()

            # default: 30
            if self.config.losses.get("resnet_pl", {"weight": 0})['weight'] > 0:
                self.loss_resnet_pl = ResNetPL(**self.config.losses.resnet_pl)
            else:
                self.loss_resnet_pl = None

        self.visualize_each_iters = visualize_each_iters
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        LOGGER.info('BaseInpaintingTrainingModule init done')

    def configure_optimizers(self):
        discriminator_params = list(self.discriminator.parameters())

        if ("lr" in self.config.optimizers.generator or
                "lr" in self.config.optimizers.discriminator):
            print("### WARNING: Changed behavior to rely on base_learning_rate.")
            assert self.learning_rate > 0

        generator_lr = self.config.optimizers.generator.pop("lr_factor", 1.0)*self.learning_rate
        self.config.optimizers.generator["lr"] = generator_lr
        print(f"Generator LR: {generator_lr}")
        discriminator_lr = self.config.optimizers.discriminator.pop("lr_factor", 1.0)*self.learning_rate
        self.config.optimizers.discriminator["lr"] = discriminator_lr
        print(f"Discriminator LR: {discriminator_lr}")

        optimizers = [
            dict(optimizer=make_optimizer(self.generator.parameters(),
                                          **self.config.optimizers.generator)),
            dict(optimizer=make_optimizer(discriminator_params,
                                          **self.config.optimizers.discriminator)),
        ]

        if self.use_scheduler:
            for i in range(len(optimizers)):
                print(f"Setting up LambdaLR scheduler {i}...")

                scheduler = instantiate_from_config(self.scheduler_config)
                scheduler = {
                    'scheduler': LambdaLR(optimizers[i]["optimizer"],
                                          lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }
                optimizers[i]["lr_scheduler"] = scheduler

        return optimizers


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


    def prepare_batch(self, batch, to_device=False):
        # NOTE our loaders deliver in bhwc [-1,1], lama uses bchw [0,1]
        for key in ["image", "mask"]:
            batch[key] = (batch[key].permute(0, 3, 1, 2).to(
                memory_format=torch.contiguous_format).float()+1.0)/2.0
            if to_device and hasattr(batch[key], "to"):
                batch[key] = batch[key].to(device=self.device)

        # bthwc [-1,1] -> btchw [0,1]
        for key in ["srcs", "srcs_masks"]:
            batch[key] = (batch[key].permute(0, 1, 4, 2, 3).to(
                memory_format=torch.contiguous_format).float()+1.0)/2.0
            if to_device and hasattr(batch[key], "to"):
                batch[key] = batch[key].to(device=self.device)


    def log_images(self, batch, **kwargs):
        self.prepare_batch(batch, to_device=True)
        batch = self(batch)

        logkeys = ["image", "predicted_image", "inpainted"]
        log = dict()
        for k in logkeys:
            log[k] = batch[k]

        batch["masked_srcs"] = batch["srcs"] * (1-batch["srcs_masks"])
        logkeys = ["masked_srcs"]
        if self.flow is not None:
            logkeys += ["aligned_srcs", "flow_errs"]
        for k in logkeys:
            for t in range(batch[k].shape[1]):
                log[f"{k}_{t}"] = batch[k][:, t]

        overviewkeys = ["image", "predicted_image", "inpainted"]
        rescale_keys = []
        if self.config.losses.adversarial.weight > 0:
            if self.store_discr_outputs_for_vis:
                overviewkeys = ["image", "predicted_image", "discr_output_fake",
                                "discr_output_real", "inpainted"]
                rescale_keys = ["discr_output_fake", "discr_output_real"]
                with torch.no_grad():
                    self.store_discr_outputs(batch)

        overview = visualize_mask_and_images_batch(batch, overviewkeys,
                                                   rescale_keys=rescale_keys,
                                                   max_items=10,
                                                   last_without_mask=True)
        log["overview"] = torch.tensor(overview)[None,...].permute(0, 3, 1, 2)

        for k in log:
            log[k] = log[k]*2.0-1.0

        if self.visualize_attn:
            attns = batch["attentions"]
            for i in range(len(attns)):
                N = 4
                C = 8*8
                read_attn = attns[i][0][:N, :C]
                read_attn = rearrange(read_attn, "b c h w -> b 1 (c h) w")
                write_attn = attns[i][1][:N, :C]
                write_attn = rearrange(write_attn, "b c h w -> b 1 (c h) w")
                log[f"read_{i}"] = read_attn*2.0-1.0
                log[f"write_{i}"] = write_attn*2.0-1.0

        return log


    def training_step(self, batch, batch_idx, optimizer_idx=None):
        self.log("global_step", self.global_step,
                 prog_bar=False, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            for i, opt in enumerate(self.optimizers()):
                lr = opt.param_groups[0]['lr']
                self.log(f'lr_abs_{i}',lr,prog_bar=False,logger=True,on_step=True,on_epoch=False)

        self.prepare_batch(batch)
        self._is_training_step = True
        return self._do_step(batch, batch_idx, mode='train', optimizer_idx=optimizer_idx)


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.prepare_batch(batch)
        self._is_training_step = False
        return self._do_step(batch, batch_idx, mode="val",
                             dataloader_idx=dataloader_idx)


    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.prepare_batch(batch)
        self._is_training_step = False
        return self._do_step(batch, batch_idx, mode="val",
                             dataloader_idx=dataloader_idx)


    def training_step_end(self, batch_parts_outputs):
        if self.training and self.average_generator \
                and self.global_step >= self.average_generator_start_step \
                and self.global_step >= self.last_generator_averaging_step + self.average_generator_period:
            if self.generator_average is None:
                self.generator_average = copy.deepcopy(self.generator)
            else:
                update_running_average(self.generator_average, self.generator, decay=self.generator_avg_beta)
            self.last_generator_averaging_step = self.global_step

        full_loss = (batch_parts_outputs['loss'].mean()
                     if torch.is_tensor(batch_parts_outputs['loss'])  # loss is not tensor when no discriminator used
                     else torch.tensor(batch_parts_outputs['loss']).float().requires_grad_(True))
        log_info = {k: v.mean() for k, v in batch_parts_outputs['log_info'].items()}
        self.log_dict(log_info, on_step=True, on_epoch=False)
        return full_loss


    def evaluator_epoch_end(self, outputs, key, evaluator):
        log_monitor_value = hasattr(self, "monitor") and self._monitor.startswith(key)
        if self.global_rank==0:
            val_evaluator_states = [s[key] for s in outputs]
            val_evaluator_res = evaluator.evaluation_end(
                states=val_evaluator_states)

            val_evaluator_res_df = pd.DataFrame(val_evaluator_res).stack(1).unstack(0)
            val_evaluator_res_df.dropna(axis=1, how='all', inplace=True)
            LOGGER.info(f'Validation metrics from {key} after epoch #{self.current_epoch}, '
                        f'total {self.global_step} iterations:\n{val_evaluator_res_df}')

            # be sure to pass rank_zero_only=True to avoid deadlock
            for k, v in flatten_dict(val_evaluator_res).items():
                self.log(f'{key}_{k}', v, rank_zero_only=True)

                if log_monitor_value and f'{key}_{k}'==self._monitor:
                    monitor_value = v
        else:
            for sc in evaluator.scores.values():
                sc.reset()
            monitor_value = 0

        if log_monitor_value:
            # must log them on all ranks or we will get misconfiguration error
            # about missing monitor value
            self.log(self.monitor, monitor_value)



    def validation_epoch_end(self, outputs):
        result = super().validation_epoch_end(outputs)

        per_data_outputs = outputs
        if type(per_data_outputs[0]) != list:
            per_data_outputs = [per_data_outputs]

        for dataloader_idx, outputs in enumerate(per_data_outputs):
            # gather evaluator results from all gpus
            outputs = [dict((k, output[k]) for k in output.keys() if
                            k.startswith("val_evaluator")) for output in outputs]
            outputs = self.all_gather(outputs)
            def reshapethem(x):
                xshape = x.shape
                outshape = (xshape[0]*xshape[1],*xshape[2:])
                return x.reshape(outshape)
            outputs = apply_to_collection(outputs, torch.Tensor, reshapethem)

            val_key = self.validation_keys[dataloader_idx]
            self.evaluator_epoch_end(
                outputs, "val_evaluator_raw_state_{}".format(val_key),
                getattr(self, "val_evaluator_raw_{}".format(val_key)))
            self.evaluator_epoch_end(
                outputs, "val_evaluator_stitched_state_{}".format(val_key),
                getattr(self, "val_evaluator_stitched_{}".format(val_key)))

        return result


    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)


    def _do_step(self, batch, batch_idx, mode='train', optimizer_idx=None,
                 dataloader_idx=None):
        if optimizer_idx == 0:  # step for generator
            set_requires_grad(self.generator, True)
            set_requires_grad(self.discriminator, False)
        elif optimizer_idx == 1:  # step for discriminator
            set_requires_grad(self.generator, False)
            set_requires_grad(self.discriminator, True)

        batch = self(batch)

        total_loss = 0
        metrics = {}

        if optimizer_idx is None or optimizer_idx == 0:  # step for generator
            total_loss, metrics = self.generator_loss(batch)

        elif optimizer_idx is None or optimizer_idx == 1:  # step for discriminator
            if self.config.losses.adversarial.weight > 0:
                total_loss, metrics = self.discriminator_loss(batch)

        metrics_prefix = f'{mode}_'
        result = dict(loss=total_loss, log_info=add_prefix_to_keys(metrics, metrics_prefix))
        if mode == 'val':
            val_key = self.validation_keys[dataloader_idx]
            log_keys = ["image", "mask", "predicted_image", "inpainted"]
            eval_batch = dict((k, batch[k]) for k in log_keys)

            # log the first batches explicitly to make sure things are ok
            if batch_idx<5 and self.global_rank==0:
                root = os.path.join(self.logdir, "images",
                                    "inpaint_eval_{}".format(val_key))
                for k in eval_batch:
                    grid = torchvision.utils.make_grid(eval_batch[k], nrow=4)
                    grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
                    grid = grid.cpu().numpy()
                    grid = (grid*255).astype(np.uint8)
                    filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                        k,
                        self.global_step,
                        self.current_epoch,
                        batch_idx)
                    path = os.path.join(root, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    Image.fromarray(grid).save(path)

            # process batches for evaluation
            val_evaluator_raw = getattr(self,
                                        "val_evaluator_raw_{}".format(val_key))
            result['val_evaluator_raw_state_{}'.format(val_key)] = val_evaluator_raw.process_batch(eval_batch)
            val_evaluator_stitched = getattr(self,
                                        "val_evaluator_stitched_{}".format(val_key))
            result['val_evaluator_stitched_state_{}'.format(val_key)] = val_evaluator_stitched.process_batch(eval_batch)

        return result

    def get_current_generator(self, no_average=False):
        if not no_average and not self.training and self.average_generator and self.generator_average is not None:
            return self.generator_average
        return self.generator

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Pass data through generator and obtain at leas 'predicted_image' and 'inpainted' keys"""
        raise NotImplementedError()

    def generator_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def discriminator_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def store_discr_outputs(self, batch):
        out_size = batch['image'].shape[2:]
        discr_real_out, _ = self.discriminator(batch['image'])
        discr_fake_out, _ = self.discriminator(batch['predicted_image'])
        batch['discr_output_real'] = F.interpolate(discr_real_out, size=out_size, mode='nearest')
        batch['discr_output_fake'] = F.interpolate(discr_fake_out, size=out_size, mode='nearest')
        batch['discr_output_diff'] = batch['discr_output_real'] - batch['discr_output_fake']

    def get_ddp_rank(self):
        return self.trainer.global_rank if (self.trainer.num_nodes * self.trainer.num_processes) > 1 else None


###### training/trainers/default.py


def make_constant_area_crop_batch(batch, **kwargs):
    crop_y, crop_x, crop_height, crop_width = make_constant_area_crop_params(img_height=batch['image'].shape[2],
                                                                             img_width=batch['image'].shape[3],
                                                                             **kwargs)
    batch['image'] = batch['image'][:, :, crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
    batch['mask'] = batch['mask'][:, :, crop_y: crop_y + crop_height, crop_x: crop_x + crop_width]
    return batch


class DefaultInpaintingTrainingModule(BaseInpaintingTrainingModule):
    def __init__(self, *args, concat_mask=True, rescale_scheduler_kwargs=None,
                 image_to_discriminator='predicted_image',
                 add_noise_kwargs=None, noise_fill_hole=False, const_area_crop_kwargs=None,
                 distance_weighter_kwargs=None, distance_weighted_mask_for_discr=False,
                 fake_fakes_proba=0, fake_fakes_generator_kwargs=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.concat_mask = concat_mask
        self.rescale_size_getter = get_ramp(**rescale_scheduler_kwargs) if rescale_scheduler_kwargs is not None else None
        self.image_to_discriminator = image_to_discriminator
        self.add_noise_kwargs = add_noise_kwargs
        self.noise_fill_hole = noise_fill_hole
        self.const_area_crop_kwargs = const_area_crop_kwargs
        self.refine_mask_for_losses = make_mask_distance_weighter(**distance_weighter_kwargs) \
            if distance_weighter_kwargs is not None else None
        self.distance_weighted_mask_for_discr = distance_weighted_mask_for_discr

        self.fake_fakes_proba = fake_fakes_proba
        if self.fake_fakes_proba > 1e-3:
            self.fake_fakes_gen = FakeFakesGenerator(**(fake_fakes_generator_kwargs or {}))

    def forward(self, batch):
        if self.training and self.rescale_size_getter is not None:
            raise NotImplementedError()
            cur_size = self.rescale_size_getter(self.global_step)
            batch['image'] = F.interpolate(batch['image'], size=cur_size, mode='bilinear', align_corners=False)
            batch['mask'] = F.interpolate(batch['mask'], size=cur_size, mode='nearest')

        if self.training and self.const_area_crop_kwargs is not None:
            raise NotImplementedError()
            batch = make_constant_area_crop_batch(batch, **self.const_area_crop_kwargs)

        img = batch['image']
        mask = batch['mask']

        masked_img = img * (1 - mask)

        srcs = batch["srcs"]
        srcs_masks = batch["srcs_masks"]

        masked_srcs = srcs * (1 - srcs_masks)

        if self.flow is not None:
            if not "flows" in batch:
                with torch.no_grad():
                    target_rgba = torch.cat((img, mask), dim=1)
                    source_rgba = torch.cat((srcs, srcs_masks), dim=2)
                    flows = list()
                    flow_errs = list()
                    aligned_srcs = list()
                    for i in range(source_rgba.shape[1]):
                        flow, flow_err = self.flow(target_rgba, source_rgba[:,i])
                        flows.append(flow)
                        flow_errs.append(flow_err)
                        aligned_srcs.append(
                            resized_flow_sample(source_rgba[:,i], flow,
                                                align_corners=True)
                        )
                    flows = torch.stack(flows, dim=1)
                    flow_errs = torch.stack(flow_errs, dim=1)
                    aligned_srcs = torch.stack(aligned_srcs, dim=1)
                    batch["flows"] = flows
                    batch["flow_errs"] = flow_errs
                    batch["aligned_srcs"] = aligned_srcs

            flows = batch["flows"]
            flow_errs = batch["flow_errs"]
        else:
            flows = None
            flow_errs = None

        if self.add_noise_kwargs is not None:
            raise NotImplementedError()
            noise = make_multiscale_noise(masked_img, **self.add_noise_kwargs)
            if self.noise_fill_hole:
                masked_img = masked_img + mask * noise[:, :masked_img.shape[1]]
            masked_img = torch.cat([masked_img, noise], dim=1)

        if self.concat_mask:
            masked_img = torch.cat([masked_img, mask], dim=1)
            masked_srcs = torch.cat([masked_srcs, srcs_masks], dim=2)

        out = self.generator(masked_img, return_attn=self.visualize_attn,
                             srcs=masked_srcs, flows=flows, flow_errs=flow_errs)
        if self.visualize_attn:
            batch["predicted_image"] = out[0]
            batch["attentions"] = out[1]
        else:
            batch['predicted_image'] = out
        batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']

        if self.fake_fakes_proba > 1e-3:
            if self.training and torch.rand(1).item() < self.fake_fakes_proba:
                batch['fake_fakes'], batch['fake_fakes_masks'] = self.fake_fakes_gen(img, mask)
                batch['use_fake_fakes'] = True
            else:
                batch['fake_fakes'] = torch.zeros_like(img)
                batch['fake_fakes_masks'] = torch.zeros_like(mask)
                batch['use_fake_fakes'] = False

        batch['mask_for_losses'] = self.refine_mask_for_losses(img, batch['predicted_image'], mask) \
            if self.refine_mask_for_losses is not None and self.training \
            else mask

        return batch

    def generator_loss(self, batch):
        img = batch['image']
        predicted_img = batch[self.image_to_discriminator]
        original_mask = batch['mask']
        supervised_mask = batch['mask_for_losses']

        # L1
        l1_value = masked_l1_loss(predicted_img, img, supervised_mask,
                                  self.config.losses.l1.weight_known,
                                  self.config.losses.l1.weight_missing)

        total_loss = l1_value
        metrics = dict(gen_l1=l1_value)

        # vgg-based perceptual loss
        if self.config.losses.perceptual.weight > 0:
            pl_value = self.loss_pl(predicted_img, img, mask=supervised_mask).sum() * self.config.losses.perceptual.weight
            total_loss = total_loss + pl_value
            metrics['gen_pl'] = pl_value

        # discriminator
        # adversarial_loss calls backward by itself
        mask_for_discr = supervised_mask if self.distance_weighted_mask_for_discr else original_mask
        self.adversarial_loss.pre_generator_step(real_batch=img, fake_batch=predicted_img,
                                                 generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(img)
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
        adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(real_batch=img,
                                                                         fake_batch=predicted_img,
                                                                         discr_real_pred=discr_real_pred,
                                                                         discr_fake_pred=discr_fake_pred,
                                                                         mask=mask_for_discr)
        total_loss = total_loss + adv_gen_loss
        metrics['gen_adv'] = adv_gen_loss
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))

        # feature matching
        # default: 100
        if self.config.losses.feature_matching.weight > 0:
            need_mask_in_fm = OmegaConf.to_container(self.config.losses.feature_matching).get('pass_mask', False)
            mask_for_fm = supervised_mask if need_mask_in_fm else None
            fm_value = feature_matching_loss(discr_fake_features, discr_real_features,
                                             mask=mask_for_fm) * self.config.losses.feature_matching.weight
            total_loss = total_loss + fm_value
            metrics['gen_fm'] = fm_value

        if self.loss_resnet_pl is not None:
            resnet_pl_value = self.loss_resnet_pl(predicted_img, img)
            total_loss = total_loss + resnet_pl_value
            metrics['gen_resnet_pl'] = resnet_pl_value

        return total_loss, metrics

    def discriminator_loss(self, batch):
        total_loss = 0
        metrics = {}

        predicted_img = batch[self.image_to_discriminator].detach()
        self.adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=predicted_img,
                                                     generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(batch['image'])
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
        adv_discr_loss, adv_metrics = self.adversarial_loss.discriminator_loss(real_batch=batch['image'],
                                                                               fake_batch=predicted_img,
                                                                               discr_real_pred=discr_real_pred,
                                                                               discr_fake_pred=discr_fake_pred,
                                                                               mask=batch['mask'])
        total_loss = total_loss + adv_discr_loss
        metrics['discr_adv'] = adv_discr_loss
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))


        if batch.get('use_fake_fakes', False):
            fake_fakes = batch['fake_fakes']
            self.adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=fake_fakes,
                                                         generator=self.generator, discriminator=self.discriminator)
            discr_fake_fakes_pred, _ = self.discriminator(fake_fakes)
            fake_fakes_adv_discr_loss, fake_fakes_adv_metrics = self.adversarial_loss.discriminator_loss(
                real_batch=batch['image'],
                fake_batch=fake_fakes,
                discr_real_pred=discr_real_pred,
                discr_fake_pred=discr_fake_fakes_pred,
                mask=batch['mask']
            )
            total_loss = total_loss + fake_fakes_adv_discr_loss
            metrics['discr_adv_fake_fakes'] = fake_fakes_adv_discr_loss
            metrics.update(add_prefix_to_keys(fake_fakes_adv_metrics, 'adv_'))

        return total_loss, metrics
