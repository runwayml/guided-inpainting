import os
import argparse
import numpy as np
from gi.modules.RAFT.raft import RAFT
import torch
from tqdm import tqdm
import kornia


def get_dilated_and_closed_mask(masks):
    with torch.no_grad():
        masks_big = masks
        kernel = torch.tensor(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            dtype=masks_big.dtype,
            device=masks_big.device,
        )
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        for _ in range(15):
            # for binary case we can do with single channel conv
            masks_big = torch.nn.functional.conv2d(masks_big, kernel, padding=1).clamp(
                0, 1
            )
        # close
        for _ in range(21):
            # fast binary dilation
            masks_big = torch.nn.functional.conv2d(masks_big, kernel, padding=1).clamp(
                0, 1
            )
            # fast binary erosion
            masks_big = (
                torch.nn.functional.conv2d(
                    torch.nn.functional.pad(
                        masks_big, (1, 1, 1, 1), mode="constant", value=1
                    ),
                    kernel,
                    padding=0,
                )
                - (kernel.sum() - 1)
            ).clamp(0, 1)
        return masks_big


def split_in_batches(iterator, n):
    out = []
    for elem in iterator:
        out.append(elem)
        if len(out) == n:
            yield out
            out = []
    if len(out) > 0:
        yield out


def generate_flows(model, frames, verbose=False):
    batch_size = 4
    n_it = 50

    with torch.inference_mode():
        frames = frames*255.0

        flows_f = []
        flows_b = []

        indices = list(range(frames.shape[0] - 1))

        iterator = split_in_batches(indices, batch_size)
        if verbose:
            iterator = tqdm(iterator)
        for indices in iterator:
            frames_1 = frames[indices, ...]
            frames_2 = frames[[i + 1 for i in indices], ...]

            _, flo_f_batch = model(frames_1, frames_2, iters=n_it, test_mode=True)
            _, flo_b_batch = model(frames_2, frames_1, iters=n_it, test_mode=True)

            flows_f.append(flo_f_batch)
            flows_b.append(flo_b_batch)

        flows_f = torch.cat(flows_f, dim=0)
        flows_b = torch.cat(flows_b, dim=0)

    return torch.stack((flows_f, flows_b))


def generate_mesh_grid(H, W, device, dtype, yx=False):
    xx = torch.arange(0, W, device=device, dtype=dtype).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=device, dtype=dtype).view(-1, 1).repeat(1, W)
    xx = xx.view(1, H, W)
    yy = yy.view(1, H, W)
    if not yx:
        grid = torch.cat((xx, yy), 0)
    else:
        grid = torch.cat((yy, xx), 0)
    return grid


def inpaint_flows(flows, masks, n_samples=20, batch_size=16, verbose=False):
    b, c, h, w = flows.shape
    indices = list(range(b))
    out = list()
    iterator = split_in_batches(indices, batch_size)
    if verbose:
        iterator = tqdm(iterator)
    for indices in iterator:
        out.append(_inpaint_flows(flows[indices], masks[indices],
                                  n_samples=n_samples))
    out = torch.cat(out, dim=0)
    return out


def _inpaint_flows(flows, masks, n_samples=20):
    outdevice = flows.device
    b, c, h, w = flows.shape
    dtype = torch.float32
    device = torch.device("cuda")
    flows = flows.to(device=device)

    masks = masks.to(dtype=torch.float32, device=flows.device)
    kernel = torch.ones(3, 3).to(masks)
    kernel[0, 0] = 0
    kernel[-1, 0] = 0
    kernel[0, -1] = 0
    kernel[-1, -1] = 0
    boundary = (masks > 0.5) & (kornia.morphology.erosion(masks, kernel) <= 0.5)
    boundary[:, :, 0, :] = 0
    boundary[:, :, -1, :] = 0
    boundary[:, :, :, -1] = 0
    boundary[:, :, :, 0] = 0

    bpoints = list()
    bvalues = list()
    bpoints_all = list()
    for i in range(b):
        nz = boundary[i, 0].nonzero()
        if nz.shape[0] == 0:
            nz = torch.tensor([[0, 0]]).to(nz)
        bpoints_all.append(nz)
        prng = np.random.default_rng(1)
        nz = nz[prng.choice(nz.shape[0], n_samples)]
        bpoints.append(nz)
        bvalues.append(flows[i, :, nz[:, 0], nz[:, 1]])
    # parallel over n_samples
    bpoints = torch.stack(bpoints)
    mg = generate_mesh_grid(h, w, device, dtype, yx=True)
    bvalues = torch.stack(bvalues)
    # bnhw
    dist = (
        mg[None, None][:, :, 0, :, :] - bpoints[:, :, :, None, None][:, :, 0, :, :]
    ) ** 2 + (
        mg[None, None][:, :, 1, :, :] - bpoints[:, :, :, None, None][:, :, 1, :, :]
    ) ** 2
    denom = dist ** 3
    denom[denom == 0] = 1e-10
    weight = 1.0 / denom
    acc = torch.sum(
        # weight: bnhw
        # bvalues: b2n
        # *: b2nhw
        # **: b2hw
        weight[:, None, :, :, :] * bvalues[:, :, :, None, None],
        2,
    )
    acc_weight = torch.sum(weight, 1)
    interp = acc / acc_weight[:, None, :, :]

    flo_filled_gpu = (1 - masks) * flows + masks * interp

    return flo_filled_gpu.to(device=outdevice)


def combine_flows(grid, flow1, flow2):
    C, H, W = flow1.shape

    grid_flow = grid + flow1
    grid_flow[0, :, :] = 2.0 * grid_flow[0, :, :] / max(W - 1, 1) - 1.0
    grid_flow[1, :, :] = 2.0 * grid_flow[1, :, :] / max(H - 1, 1) - 1.0
    grid_flow = grid_flow.permute(1, 2, 0)
    flow2_inter = torch.nn.functional.grid_sample(
        flow2[None, ...], grid_flow[None, ...], mode="nearest", align_corners=True
    )[0]

    return flow1 + flow2_inter


def initialize_RAFT(checkpoint_path, small=False, to_cuda=True):
    args = argparse.Namespace()
    args.small = small
    args.mixed_precision = True
    args.dropout = 0
    args.alternate_corr = False

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    model = model.module
    if to_cuda:
        model = model.cuda()
    model = model.eval()

    return model


def get_inpainted_flow(frames, masks, cache_path=None,
                       raft_path="checkpoints/raft-things.pth"):
    if cache_path is not None and os.path.exists(cache_path):
        return torch.load(cache_path)

    masks = get_dilated_and_closed_mask(masks)

    raft = initialize_RAFT(checkpoint_path=raft_path)
    flows_f, flows_b = generate_flows(raft, frames)
    flows_f = inpaint_flows(flows_f, masks[:-1])
    flows_b = inpaint_flows(flows_b, masks[1:])

    flows = torch.stack([flows_f, flows_b], dim=0)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(flows, cache_path)
    return flows


def get_grid(H, W, dtype, device):
    xx = torch.arange(0, W, device=device, dtype=dtype).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=device, dtype=dtype).view(-1, 1).repeat(1, W)
    xx = xx.view(1, H, W)
    yy = yy.view(1, H, W)
    grid = torch.cat((xx, yy), 0)
    return grid


def get_factor(H, W, dtype, device):
    return (
        torch.tensor(
            [2.0 / max(W - 1, 1), 2.0 / max(H - 1, 1)], dtype=dtype, device=device
        )
        .unsqueeze(0)
        .unsqueeze(2)
        .unsqueeze(3)
    )


def flow_to_grid(flo):
    assert flo.dtype == torch.float32, "Half precision not sufficient"

    _, H, W = flo.size()
    grid = get_grid(H, W, dtype=flo.dtype, device=flo.device)
    factor = get_factor(H, W, dtype=flo.dtype, device=flo.device)

    vgrid = grid + flo.unsqueeze(0)  # bc
    vgrid = factor * vgrid - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    return vgrid[0]


def flows_to_grids(flo):
    assert flo.dtype == torch.float32, "Half precision not sufficient"

    B, _, H, W = flo.size()
    grid = get_grid(H, W, dtype=flo.dtype, device=flo.device)
    factor = get_factor(H, W, dtype=flo.dtype, device=flo.device)

    vgrid = grid + flo
    vgrid = factor * vgrid - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    return vgrid


def flow_sample(input, flow, mode="bilinear", padding_mode="zeros", align_corners=None):
    # wraps a flow sample method until it becomes available natively
    # currently has to operate via grid sample which requires a normalized grid
    # for which half precision is not good enough
    input_dtype = input.dtype
    grid_dtype = torch.float32

    b, c, h, w = input.shape
    b_, c_, h_, w_ = flow.shape
    assert b == b_

    grid = flows_to_grids(flow.to(dtype=grid_dtype))

    return torch.nn.functional.grid_sample(
        input=input.to(dtype=grid_dtype),
        grid=grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    ).to(input_dtype)


def resize_flows(flows, hw):
    orig_hw = flows.shape[-2:]
    if orig_hw == hw:
        return flows
    flows = torch.nn.functional.interpolate(
        flows, size=hw, mode="bilinear", align_corners=True
    )
    alpha_y = hw[0] / orig_hw[0]
    alpha_x = hw[1] / orig_hw[1]
    flows[:, 0, ...] *= alpha_x
    flows[:, 1, ...] *= alpha_y
    return flows


def resized_flow_sample(input, flow, mode="bilinear", padding_mode="zeros", align_corners=None):
    input_dtype = input.dtype
    grid_dtype = torch.float32

    b, c, h, w = input.shape
    b_, c_, h_, w_ = flow.shape
    assert b == b_

    flow = resize_flows(flow, (h, w))
    grid = flows_to_grids(flow.to(dtype=grid_dtype))

    return torch.nn.functional.grid_sample(
        input=input.to(dtype=grid_dtype),
        grid=grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    ).to(input_dtype)



def normalize_grid(grid):
    assert grid.dtype == torch.float32, "Half precision not sufficient"

    _, H, W = grid.size()
    factor = get_factor(H, W, dtype=grid.dtype, device=grid.device)

    vgrid = grid
    vgrid = factor * vgrid - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    return vgrid[0]


def pgrid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=None):
    # wraps a grid sample method until it becomes available natively
    # currently has to operate via grid sample which requires a normalized grid
    # for which half precision is not good enough
    input_dtype = input.dtype
    grid_dtype = torch.float32

    b, c, h, w = input.shape
    b_, c_, h_, w_ = grid.shape
    assert b == b_
    if b_ != 1:
        raise NotImplementedError()

    grid = normalize_grid(grid[0].to(dtype=grid_dtype)).unsqueeze(0)

    return torch.nn.functional.grid_sample(
        input=input.to(dtype=grid_dtype),
        grid=grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    ).to(input_dtype)


def extend_flows(flows, HW, offset):
    H, W = HW
    b, c, h, w = flows.shape
    device = flows.device
    dtype = flows.dtype
    bigflow = torch.zeros((b, c, H, W), device=device, dtype=dtype)
    bigmask = torch.ones((b, c, H, W), device=device, dtype=dtype)
    bigflow[:, :, offset[0]:offset[0]+h, offset[1]:offset[1]+w] = flows
    bigmask[:, :, offset[0]+1:offset[0]+h-1, offset[1]+1:offset[1]+w-1] = 0.0
    return inpaint_flows(bigflow, bigmask, batch_size=1)
