import argparse, os, sys, datetime, glob, importlib, csv
from packaging import version
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset, Subset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.trainer.supporters import CombinedLoader
from einops import rearrange
import pandas as pd
import logging
import imageio

try:
    from pytorch_lightning.plugins import DeepSpeedPlugin
except ImportError:
    DeepSpeedPlugin = None # no xtra speed for you
from pytorch_lightning.utilities import rank_zero_info
from matplotlib import pyplot as plt


from gi.util import create_or_append


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-x",
        "--xtraspeed",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable use of the deepspeed",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )

    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size=None, train=None, validation=None, test=None,
                 predict=None, wrap=False, num_workers=None,
                 use_worker_init_fn=False,
                 batch_size_validation=None,
                 batch_size_test=None,
                 batch_size_predict=None,
                 combined_val_batches=True,
                 combined_test_batches=True):
        super().__init__()
        self.batch_size = batch_size
        self.batch_sizes = {"train": self.batch_size,
                            "validation": batch_size_validation,
                            "test": batch_size_test,
                            "predict": batch_size_predict}
        self.num_workers = num_workers
        self.use_worker_init_fn = use_worker_init_fn
        self.wrap = wrap
        self.combined_val_batches = combined_val_batches
        self.combined_test_batches = combined_test_batches

        splits = dict()
        if train is not None:
            splits["train"] = train
        if validation is not None:
            splits["validation"] = validation
        if test is not None:
            splits["test"] = test
        if predict is not None:
            splits["predict"] = predict

        self.dataset_configs = dict()
        for split in splits:
            configs = {"default": splits[split]} if "target" in splits[split] else splits[split]
            total_batch_size = 0
            for k in configs:
                self.dataset_configs[split+"/"+k] = configs[k]
                self.set_defaults(split, self.dataset_configs[split+"/"+k], k)
                total_batch_size += self.dataset_configs[split+"/"+k]["batch_size"]
            if split == "train":
                self.batch_size = total_batch_size
            loader_name = split+"_dataloader" if split!="validation" else "val_dataloader"
            setattr(self, loader_name, getattr(self, "_"+loader_name))


    def set_defaults(self, split, config, datakey):
        config["batch_size"] = config.get("batch_size",
                                          self.batch_sizes[split] or self.batch_size)
        if config["batch_size"] is None:
            raise ValueError(f"Could not determine batch size for {split+'/'+datakey}")
        default_num_workers = self.num_workers if self.num_workers is not None else min(8, 2*config["batch_size"])
        config["num_workers"] = config.get("num_workers", self.num_workers) or default_num_workers
        default_shuffle = True if split=="train" else False
        config["shuffle"] = config.get("shuffle", default_shuffle)
        config["use_worker_init_fn"] = config.get("use_worker_init_fn",
                                                  self.use_worker_init_fn)


    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)


    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])


    def _generic_dataloader(self, split):
        datakeys = [x.split("/", maxsplit=1)[1] for x in self.datasets if
                    x.startswith(split)]
        dataloaders = {k: DataLoader(
            self.datasets[split+"/"+k],
            batch_size=self.dataset_configs[split+"/"+k]["batch_size"],
            num_workers=self.dataset_configs[split+"/"+k]["num_workers"],
            shuffle=self.dataset_configs[split+"/"+k]["shuffle"],
            worker_init_fn=(self.worker_init_fn if
                            self.dataset_configs[split+"/"+k]["use_worker_init_fn"] else
                            None),
        ) for k in datakeys}
        if len(dataloaders) == 1 and "default" in dataloaders:
            return dataloaders["default"]
        return dataloaders


    def _train_dataloader(self):
        return self._generic_dataloader("train")


    def _val_dataloader(self):
        loaders = self._generic_dataloader("validation")
        if type(loaders)==dict:
            if self.combined_val_batches:
                loaders = CombinedLoader(loaders, "max_size_cycle")
            else:
                loaders = [loaders[k] for k in sorted(loaders.keys())]
        return loaders


    def _test_dataloader(self):
        loaders = self._generic_dataloader("test")
        if type(loaders)==dict:
            if self.combined_test_batches:
                loaders = CombinedLoader(loaders, "max_size_cycle")
            else:
                loaders = [loaders[k] for k in sorted(loaders.keys())]
        return loaders


    def _predict_dataloader(self):
        return self._generic_dataloader("predict")



class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, txtdir, config,
                 lightning_config, debug):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.txtdir = txtdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.debug = debug

    def on_keyboard_interrupt(self, trainer, pl_module):
        if (not self.debug) and trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.txtdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir,'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

            logging.basicConfig(filename=os.path.join(self.txtdir, "{}.log".format(self.now)),
                                level=logging.INFO, force=True)

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

    def on_test_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs and setup logging
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.txtdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

            logging.basicConfig(filename=os.path.join(self.txtdir, "{}.log".format(self.now)),
                                level=logging.INFO, force=True)


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False,
                 log_on_batch_idx=True,log_images_kwargs=None,
                 log_first_idx=False):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_idx = log_first_idx

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        raise ValueError("No way wandb")
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids)

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            as_vid = False
            if len(images[k].shape)==5:
                if images[k].shape[1] < 8:
                    images[k] = rearrange(images[k], "b t c h w -> b c (t h) w")
                else:
                    as_vid = True
            if not as_vid:
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                if self.rescale:
                    grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid*255).astype(np.uint8)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                    k,
                    global_step,
                    current_epoch,
                    batch_idx)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(grid).save(path)
            else:
                vid = images[k]
                T = vid.shape[1]
                if self.rescale:
                    vid = (vid+1.0)/2.0
                vid = rearrange(vid, "b t c h w -> t h (b w) c", t=T)
                vid = (vid.numpy()*255).astype(np.uint8)
                vid = vid[...,:4]
                if vid.shape[-1]==1:
                    vid = vid.squeeze(-1)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.mp4".format(
                    k,
                    global_step,
                    current_epoch,
                    batch_idx)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                imageio.mimsave(path, vid)


    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0 and
                (self.log_first_idx or check_idx>0)):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split,**self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_images = dict((k, images[k]) for k in images
                                 if hasattr(images[k], "shape") and
                                 len(images[k].shape)==4)
            logger_log_images(pl_module, logger_images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if (check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                #print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer,pl_module,batch_idx=batch_idx)


    @rank_zero_only
    def log_noisy_accuracies(self,trainer,pl_module):
        accs = {}
        for t in pl_module.noisy_acc:
            [create_or_append(accs,k,(t,np.mean(pl_module.noisy_acc[t][k]))) for k in pl_module.noisy_acc[t]]

        n_subplots = len(accs)
        max_n_row = 4
        n_row = min(max(n_subplots//max_n_row,1),max_n_row)
        n_col = n_subplots // n_row
        if n_subplots%n_row != 0:
            n_col +=1
        fig, axs = plt.subplots(n_row, n_col)
        plt.title("Accuracies for different noise levels")
        for ax_idx, ylabel in enumerate(accs):
            data = accs[ylabel]
            data_x = [d[0] for d in data]
            data_y = [d[1] for d in data]
            axs[ax_idx].plot(data_x,data_y)
            axs[ax_idx].set_xlabel('timestep')
            axs[ax_idx].set_ylabel(ylabel)
            # axs[ax_idx].set_xticks(data[0])
            # axs[ax_idx].set_xticklabels(str(t) for t in data[0])


        step = pl_module.global_step
        epoch = pl_module.current_epoch
        savepath = os.path.join(pl_module.logger.save_dir,'noise_level_accs')
        os.makedirs(savepath,exist_ok=True)
        savename = os.path.join(savepath,f'epoch-{epoch}_global_step-{step}.png')

        plt.tight_layout()
        fig.savefig(savename)
        plt.close()
        # reset accuracies for the next validation epoch
        pl_module.reset_noise_accs()

    @rank_zero_only
    def log_gradients(self,trainer, pl_module,is_end_epoch=False,batch_idx=None):
        from matplotlib import pyplot as plt
        step = pl_module.global_step
        epoch = pl_module.current_epoch
        # last scale has no gradients
        grad_norms = {key: pl_module.grad_norms_log[key][:-1] for key in pl_module.grad_norms_log}
        grad_norms['norm'] = [float(torch.stack(n).mean().cpu().numpy()) if len(n) > 0 else 0 for n in grad_norms['norm']]

        fig, axs = plt.subplots(1,3)

        savepath = os.path.join(pl_module.logger.save_dir,'gradient_norms'
                                )
        os.makedirs(savepath,exist_ok=True)
        # grad norms over alpha_t_bar
        axs[0].plot(grad_norms['alpha_bar'],grad_norms['norm'],color='b')
        axs[0].set_xlabel('alpha_t_cum')
        axs[0].set_ylabel('grad_norms')
        axs[0].set_title('Grad norms vs alpha_t_cum')
        axs[0].set_xticks(grad_norms['alpha_bar'])
        axs[0].set_xticklabels([str(e) for e in grad_norms['alpha_bar']],rotation='vertical')

        # grad norms over beta_t
        axs[1].plot(grad_norms['beta'], grad_norms['norm'], color='r')
        axs[1].set_xlabel('beta_t')
        axs[1].set_ylabel('grad_norms')
        axs[1].set_title('Grad norms vs beta_t')
        axs[1].set_xticks(grad_norms['beta'])
        axs[1].set_xticklabels([str(e) for e in grad_norms['beta']], rotation='vertical')

        # grad norms over t
        axs[2].plot(grad_norms['t'], grad_norms['norm'], color='g')
        axs[2].set_xlabel('t')
        axs[2].set_ylabel('grad_norms')
        axs[2].set_title('Grad norms vs t')
        axs[2].set_xticks(grad_norms['t'])
        axs[2].set_xticklabels([str(e) for e in grad_norms['t']], rotation='vertical')

        postfix = 'EPOCH_END' if is_end_epoch else f'IT_{batch_idx}'
        savename = os.path.join(savepath,f'grad_norms_gs-{step:06}_e-{epoch:06}-VAL_{postfix}.png')
        # plt.tight_layout()
        fig.savefig(savename)
        plt.close()

        # also save raw data
        df = pd.DataFrame.from_dict(pl_module.grad_norms_log)

        df.to_csv(os.path.join(os.path.join(savepath,f'grad_norms-raw_data-gs-{step:06}_e-{epoch:06}-VAL_{postfix}.csv')))

    def on_validation_end(self, trainer, pl_module):
        if hasattr(pl_module, 'log_grads'):
            if (pl_module.log_grads or pl_module.calibrate_grad_norm) and pl_module.global_step > 0:
                self.log_gradients(trainer,pl_module,is_end_epoch=True)
            if hasattr(pl_module, 'grad_norms_log'):
                    pl_module.grad_norms_log['norm'] = [[] for _ in pl_module.grad_norms_log['norm']]

        # classifier support
        if hasattr(pl_module, 'noisy_acc'):
            self.log_noisy_accuracies(trainer,pl_module)


class FIDelity(Callback):
    def __init__(
        self,
        *,
        data_cfg,
        split="validation",
        num_images=5000,
        isc=True,
        kid=True,
        fid=True,
        epoch_frequency=1,
        step_frequency=None,
        min_epochs=1,
        min_steps=0,
        input_key="inputs",
        output_key="reconstructions",
        load_input_from=None,
        save_input_to=None,
        clamp=True,
        keep_intermediate_output=False,
        log_images_kwargs=None,
        add_to_pl_metrics=True,
        **fid_kwargs
    ):
        # TODO: optionally add possibility to use cached, precomputed inceptionv3-features
        """Callback for evaluating and logging FID, IS, etc.
        Based on https://torch-fidelity.readthedocs.io/en/latest/api.html.
        .. note::
            Requires the ``log_images`` method of the respective pl-module
            to return a dict containing the ``input_key`` and ``output_key``
            keys (these are passed to the logging method as ``to_log``).
        Args:
            data_cfg: gi.main.DataModuleFromConfig configuration. Passed to
                gi.main.instantiate_from_config.
            split: dset split to use, can be one of: train, validation, test.
            num_images: Number of images contained in the dataset configured by
                ``data_cfg``. If < 0, the whole dataset split is used.
                Note that the effective number of created images depends on the
                number of images returned by the pl_module.log_images method.
            isc: Whether to calculate the Inception Score
            kid: Whether to calculate the Kernel Inception Distance
            fid: Whether to calculate the Frechet Inception Distance
            epoch_frequency: Number of epochs between score evaluations. Set to
                None to disable epoch-periodic evaluation.
            step_frequency: Number of steps between score evaluations. Set to
                None to disable step-periodic evaluation.
            min_epochs: If epoch-periodic evaluation is enabled, defines
                starting threshold.
            min_steps: If step-periodic evaluation is enabled, defines starting
                threshold.
            input_key: Input image logging key
            output_key: Output image logging key
            load_input_from: Custom path to directory containing the input
                images (e.g. previously written there via save_input_to).
            save_input_to: Custom path to directory where the input images are
                written to. May not be given together with load_input_from.
            clamp: Whether to clamp images to [0, 1]
            log_images_kwargs: Passed to pl_module.log_images
            keep_intermediate_output: Whether to store output images for each
                evaluation separately. If False, overwrites previous outputs.
            **fid_kwargs: Passed to torch_fidelity.calculate_metrics
        """
        super().__init__()
        self.data_cfg = data_cfg
        self.split = split
        self.num_images = num_images
        self.input_key = input_key
        self.output_key = output_key
        self.epoch_frequency = epoch_frequency
        self.step_frequency = step_frequency
        self.min_epochs = min_epochs
        self.min_steps = min_steps
        assert not (load_input_from is not None and save_input_to is not None)
        self.load_input_from = load_input_from
        self.save_input_to = save_input_to
        self.keep_intermediate = keep_intermediate_output

        self.isc = isc
        self.kid = kid
        self.fid = fid
        self.clamp = clamp
        self.log_images_kwargs = log_images_kwargs or {}
        self.fid_kwargs = fid_kwargs
        self.add_to_pl_metrics = add_to_pl_metrics
        if self.add_to_pl_metrics:
            print(f'Adding FIDelity computes with {self.num_images} examples to pl metrics.')
            self.epoch_frequency = 1
            self.min_epochs = 0

        self.prepared = False
        self.input_cached = False
        self.executed_steps = list()

    @rank_zero_only
    def prepare(self, logdir):
        if not self.prepared:
            self.init_data()
            self._init_folders(logdir)
            self.prepared = True

    @rank_zero_only
    def _init_folders(self, logdir):
        # set up directories where the images will be stored at
        workdir = os.path.join(logdir, "fidelity", self.dset_name)
        indir = os.path.join(workdir, self.input_key)
        outdir = os.path.join(workdir, self.output_key)

        if self.load_input_from is not None:
            indir = self.load_input_from
            if not os.path.isdir(indir):
                raise FileNotFoundError(f"Cache directory {indir} not found.")
        elif self.save_input_to is not None:
            indir = self.save_input_to

        os.makedirs(indir, exist_ok=True)
        os.makedirs(outdir, exist_ok=True)
        self.workdir = workdir
        self.indir = indir
        self.outdir = outdir

    @rank_zero_only
    def init_data(self):
        # make the dataset on which the FID will be evaluated
        data = instantiate_from_config(self.data_cfg)
        data.prepare_data()
        data.setup()
        dset = data.datasets[self.split]
        self.dset_name = dset.__class__.__name__

        if 0 <= self.num_images < len(dset):
            subindices = np.random.choice(
                np.arange(len(dset)), replace=False, size=(self.num_images,)
            )
            dset = Subset(dset, subindices)

        self.n_data = len(dset)
        self.dloader = DataLoader(
            dset,
            batch_size=data.batch_sizes[self.split],
            num_workers=data.num_workers,
            drop_last=False,
        )

    @rank_zero_only
    def log_single_img(self, img, path):
        img = (img + 1.) / 2.
        img = img.transpose(0, 1).transpose(1, 2).squeeze(-1)
        img = img.detach().cpu().numpy()
        img = (255 * img).astype(np.uint8)
        Image.fromarray(img).save(path)

    def to_callback_metrics(self,scores_dict:dict,trainer:pl.Trainer,pl_module:pl.LightningModule):
        """
        to save the calculated metrics in the dict for tracking callback metrics, which is required when intending to"
        save checkpoints which have fidelity as a monitor between two epochs
        :param scores_dict:
        :param trainer:
        :param pl_module:
        :return:
        """
        # fidelity_scores = {key: torch.Tensor([scores_dict[key]]).to(pl_module.device) for key in scores_dict}
        trainer.logged_metrics.update(scores_dict)
        trainer.callback_metrics.update(scores_dict)


    def on_batch_end(self, trainer:pl.Trainer, pl_module):
        if self.step_frequency is not None:
            if (
                pl_module.global_step % self.step_frequency == 0
                and pl_module.global_step >= self.min_steps
                and pl_module.global_step not in self.executed_steps
            ):
                self.prepare(logdir=trainer.logdir)
                fidelity_scores = self.eval_metrics(pl_module)
                self.executed_steps.append(pl_module.global_step)

                self.to_callback_metrics(fidelity_scores,trainer,pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if (self.epoch_frequency is not None or self.add_to_pl_metrics) and not trainer.sanity_checking:
            if (
                pl_module.current_epoch % self.epoch_frequency == 0
                and pl_module.current_epoch >= self.min_epochs
                and pl_module.global_step not in self.executed_steps
            ) or ( pl_module.current_epoch >= self.min_epochs and self.add_to_pl_metrics
            ) and pl_module.global_step > 0:
                self.prepare(logdir=trainer.logdir)
                fidelity_scores = self.eval_metrics(pl_module)
                self.executed_steps.append(pl_module.global_step)

                self.to_callback_metrics(fidelity_scores, trainer, pl_module)

    @rank_zero_only
    def eval_metrics(self, pl_module:pl.LightningModule):
        gs = pl_module.global_step
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        # Input data is always the same and is thus created only once
        indir = self.indir

        if self.keep_intermediate:
            outdir = os.path.join(self.outdir, f"gs-{gs:09}")
            os.mkdir(outdir) # should not overwrite anything
        else:
            outdir = self.outdir # overwrite previous data

        keys = [self.input_key, self.output_key]
        roots = {self.input_key: indir, self.output_key: outdir}

        img_count = {k: 0 for k in keys}
        for batch in tqdm(
            self.dloader,
            desc="Creating images for fidelity scores",
            leave=False,
        ):
            with torch.no_grad():
                # NOTE This requires `log_images` to accept the `to_log` kwarg.
                #      The return value should be a dict containing the
                #      input_key and output_key as keys.
                images = pl_module.log_images(
                    batch, to_log=keys, **self.log_images_kwargs
                )

            for k, save_dir in roots.items():
                if k == self.input_key and (
                    self.input_cached or self.load_input_from is not None
                ):
                    continue

                imgs = images[k]
                if self.clamp:
                    imgs = torch.clamp(imgs, -1., 1.)
                for img in imgs:
                    filepath = os.path.join(save_dir, f"{img_count[k]:06}.png")
                    self.log_single_img(img, filepath)
                    img_count[k] += 1

        from torch_fidelity import calculate_metrics
        scores = calculate_metrics(
            input1=outdir,
            input2=indir,
            isc=self.isc,
            fid=self.fid,
            kid=self.kid,
            verbose=False,
            **self.fid_kwargs
        )

        # Write scores to csv file and log them
        csv_path = os.path.join(self.workdir, "fid.csv")
        with open(csv_path, "a") as f:
            w = csv.writer(f)
            if not self.input_cached:
                # Write header lines
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                w.writerow(["timestamp", now])
                w.writerow(["keys", keys])
                w.writerow(["step", "num_samples"] + list(scores.keys()))
            w.writerow([gs, self.n_data] + list(scores.values()))

        for k, v in scores.items():
            pl_module.log(k, v, logger=True, on_epoch=True)
        if is_train:
            pl_module.train()

        self.input_cached = True # always True after first eval_metrics call

        return scores


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: gi.main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python gi.main.py`
    # (in particular `gi.main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            #idx = len(paths)-paths[::-1].index("logs")+1
            #logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs+opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_"+opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_"+cfg_name
        else:
            name = ""
        nowname = now+name+opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    txtdir = os.path.join(logdir, "logs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["accelerator"] = "ddp"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        # NOTE wandb < 0.10.0 interferes with shutdown
        # wandb >= 0.10.0 seems to fix it but still interferes with pudb
        # debugging (wrongly sized pudb ui)
        # thus prefer testtube for now
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        logger_cfg = lightning_config.get("logger") or OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        if opt.xtraspeed:
            assert (int(pl.__version__.split(".")[0]) >= 1) and (int(pl.__version__.split(".")[1]) > 0), "need lightning >=1.1 for deepspeed usage"
            # TODO: need a deepspeed-config
            print("falcon lift-off")
            trainer_kwargs["plugins"] = DeepSpeedPlugin(stage=3, cpu_offload=True)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        modelckpt_cfg = lightning_config.get("modelcheckpoint") or OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "gi.main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "txtdir": txtdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                    "debug": opt.debug,
                }
            },
            "image_logger": {
                "target": "gi.main.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "gi.main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    #"log_momentum": True
                }
            },
            "cuda_callback": {
              "target": "gi.main.CUDACallback"
            },
        }
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback':modelckpt_cfg})
        callbacks_cfg = lightning_config.get("callbacks") or OmegaConf.create()
        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            print('Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                         "filename": "{epoch:06}-{step:09}",
                         "verbose": True,
                         'save_top_k': -1,
                         'every_n_train_steps': 10000,
                         'save_weights_only': True
                     }
                     }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt,'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir  ###

        model.logdir = logdir
        print(f"Set model.logdir to {model.logdir}.")

        # data
        data = instantiate_from_config(config.data)
        data.prepare_data()
        data.setup()
        print("#### Data #####")
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # configure learning rate
        bs, base_lr = data.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1
        accumulate_grad_batches = lightning_config.trainer.get("accumulate_grad_batches") or 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
            model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb; pudb.set_trace()

        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                if not opt.debug:
                    melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            ckpt_path=None
            if not opt.train:
                ckpt_path = opt.resume_from_checkpoint
                print(f"Testing {ckpt_path}")
                state = torch.load(ckpt_path, map_location=model.device)
                missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
                print(f"Restored from {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
                if len(missing) > 0:
                    print(f"Missing Keys: {missing}")
                if len(unexpected) > 0:
                    print(f"Unexpected Keys: {unexpected}")
                if "epoch" in state:
                    model.restored_epoch = state["epoch"]
                else:
                    model.restored_epoch = 0
                del state
            else:
                print("Testing current weights.")
            trainer.test(model=model, dataloaders=data, ckpt_path=None)
    except Exception:
        if opt.debug and trainer.global_rank==0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank==0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank==0:
            print(trainer.profiler.summary())
