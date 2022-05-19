import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gi.modules.RAFT.flow_utils import initialize_RAFT, resize_flows, inpaint_flows, resized_flow_sample


class RAFT(nn.Module):
    def __init__(self, small=False, checkpoint_path=None, resolution=None,
                 return_err=False, input_range="-11", return_backward=False):
        super().__init__()
        if checkpoint_path is None:
            checkpoint_path = ("checkpoints/flow/raft/raft-things.pth" if not small
                               else"checkpoints/flow/raft/raft-small.pth")
        self.raft = initialize_RAFT(checkpoint_path=checkpoint_path,
                                    small=small, to_cuda=False)
        self.resolution = tuple(resolution) # hw
        assert len(self.resolution) == 2
        assert self.resolution[0]%8 == 0
        assert self.resolution[1]%8 == 0
        self.return_err = return_err
        self.return_backward = return_backward
        self.input_range = input_range
        assert self.input_range in ["-11", "01"]


    def forward(
        self, image1, image2, iters=20, flow_init=None, upsample=True, test_mode=True
    ):
        b, chns, h, w = image1.shape
        assert chns in [3, 4]
        if self.input_range in ["01"]:
            assert 0 <= image1.min() <= image1.max() <= 1
            image1 = 2.0*image1-1.0
            image2 = 2.0*image2-1.0

        if chns == 4:
            mask = (image1[:,3:4]+1.0)/2.0
            image1 = image1[:,:3]
            image2 = image2[:,:3]

        image1 = torch.nn.functional.interpolate(
            image1, size=self.resolution, mode="bilinear", align_corners=True
        )
        image2 = torch.nn.functional.interpolate(
            image2, size=self.resolution, mode="bilinear", align_corners=True
        )
        flow = self.raft(image1, image2,
                         iters=iters, flow_init=flow_init,
                         upsample=upsample, test_mode=test_mode,
                         is_normalized=True)[1]
        flow = resize_flows(flow, (h, w))

        if chns == 4:
            flow = inpaint_flows(flow, mask)

        if self.return_err or self.return_backward:
            flow_b = self.raft(image2, image1,
                               iters=iters, flow_init=flow_init,
                               upsample=upsample, test_mode=test_mode,
                               is_normalized=True)[1]
            flow_b = resize_flows(flow_b, (h, w))

            if chns == 4:
                flow_b = inpaint_flows(flow_b, mask)

            aligned_flow_b = resized_flow_sample(flow_b, flow, align_corners=True)
            error = torch.abs(flow + aligned_flow_b)
            alpha_y = h/self.resolution[0]
            alpha_x = w/self.resolution[1]
            error[:, 0, ...] *= alpha_x
            error[:, 1, ...] *= alpha_y
            error = error.mean(dim=1, keepdim=True)

            returns = [flow]
            if self.return_err:
                returns.append(error)

            if self.return_backward:
                returns.append(flow_b)
                aligned_flow = resized_flow_sample(flow, flow_b,
                                                   align_corners=True)

                if self.return_err:
                    error_b = torch.abs(flow_b + aligned_flow)
                    error_b[:, 0, ...] *= alpha_x
                    error_b[:, 1, ...] *= alpha_y
                    error_b = error_b.mean(dim=1, keepdim=True)
                    returns.append(error_b)

            return returns

        return flow



class Identity(nn.Module):
    """for ablating effect of alignment"""
    def __init__(self, return_err=False):
        super().__init__()
        self.return_err = return_err


    def forward(
        self, image1, image2, *args, **kwargs,
    ):
        b, chns, h, w = image1.shape
        flow = torch.zeros((b, 2, h, w), device=image1.device,
                           dtype=torch.float32)
        if self.return_err:
            err = torch.zeros((b, 1, h, w), device=image2.device,
                              dtype=torch.float32)
            return flow, err

        return flow
