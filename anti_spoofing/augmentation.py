"""Train-only appearance variation for normalized RGB batches.

The small translucent upper-face patch approximates obstruction/reflection. It
is not a photorealistic eyeglasses simulation or a guarantee of glasses
robustness. The function accepts no labels: both classes receive the same
transformation distribution. Validation and inference must not call it.
"""
from __future__ import annotations

import math

import torch


def augment_camera_batch(images: torch.Tensor) -> torch.Tensor:
    """Return a new finite NCHW RGB tensor in [-1,1], preserving device/dtype.

    Horizontal flip p=.5, independent brightness/contrast in [.75,1.25], and
    p=.2 translucent dark/bright patches of 1–6% of the image area. Patches are
    restricted to the upper face and less than half the image width. Tiny
    inputs (<16 pixels on either side) receive only photometric variation.
    All randomness uses PyTorch's seeded generator on the input device. Input
    tensors, including cached dataset tensors, are never modified in place.
    """
    if (not isinstance(images, torch.Tensor) or not images.is_floating_point()
            or images.ndim != 4 or images.shape[1] != 3 or min(images.shape) < 1):
        raise ValueError("Camera augmentation requires a nonempty floating NCHW RGB batch.")
    if not bool(torch.isfinite(images).all()) or bool(((images < -1) | (images > 1)).any()):
        raise ValueError("Camera augmentation requires finite pixels normalized to [-1,1].")
    batch, _, height, width = images.shape
    shape = (batch, 1, 1, 1)

    def rand(size=shape):
        return torch.rand(size, device=images.device, dtype=images.dtype)

    pixels = (images + 1.) * .5
    pixels = torch.where(rand() < .5, pixels.flip(-1), pixels)
    mean = pixels.mean(dim=(1, 2, 3), keepdim=True)
    pixels = ((pixels-mean) * (.75+.5*rand()) + mean) * (.75+.5*rand())
    pixels = pixels.clamp(0., 1.)

    if min(height, width) >= 16:
        # Integer bounds maintain the patch-area contract even after rounding.
        minimum_area = math.ceil(.01*height*width)
        maximum_area = math.floor(.06*height*width)
        maximum_height = max(1, math.floor(.2*height))
        minimum_width = max(1, math.ceil(minimum_area/maximum_height))
        maximum_width = max(minimum_width, math.floor(.45*width))
        area = minimum_area + rand() * (maximum_area-minimum_area)
        rectangle_width = torch.sqrt(area*(1.5+1.5*rand())).long().clamp(minimum_width, maximum_width)
        minimum_height = torch.div(minimum_area+rectangle_width-1, rectangle_width, rounding_mode="floor")
        height_limit = torch.div(maximum_area, rectangle_width, rounding_mode="floor").clamp(max=maximum_height)
        rectangle_height = (area/rectangle_width).long()
        rectangle_height = torch.minimum(torch.maximum(rectangle_height, minimum_height), height_limit)
        left = ((.25+.5*rand())*width-rectangle_width/2).long().clamp(min=0)
        left = torch.minimum(left, width-rectangle_width)
        top = ((.28+.16*rand())*height-rectangle_height/2).long().clamp(min=0)
        top = torch.minimum(top, height-rectangle_height)
        x = torch.arange(width, device=images.device).view(1, 1, 1, width)
        y = torch.arange(height, device=images.device).view(1, 1, height, 1)
        mask = ((x >= left) & (x < left+rectangle_width)
                & (y >= top) & (y < top+rectangle_height) & (rand() < .2))
        patch_value = torch.where(rand() < .5, .05+.15*rand(), .8+.15*rand())
        opacity = .25+.35*rand()
        pixels = torch.where(mask, pixels*(1-opacity)+patch_value*opacity, pixels)
    return (pixels*2.-1.).clamp(-1., 1.)


__all__ = ["augment_camera_batch"]
