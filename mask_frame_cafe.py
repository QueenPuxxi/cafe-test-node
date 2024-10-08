import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import scipy.ndimage
import numpy as np
from contextlib import nullcontext
import os

import model_management
from comfy.utils import ProgressBar
from nodes import MAX_RESOLUTION

import folder_paths

from .subtractmasks import subtract_masks

class maskframecafe:
    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": MAX_RESOLUTION,
                    "step": 1
                }),
                "incremental_expandrate": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1
                }),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "flip_input": ("BOOLEAN", {"default": False}),
                "blur_radius": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100,
                    "step": 0.1
                }),
                "lerp_alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "decay_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
            "optional": {
                "fill_holes": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "expand_mask"
    CATEGORY = "☕️不妨请咖啡哥来一杯拿铁☕️"
    DESCRIPTION = """
☕️把扩展的像素与原蒙版作差值，得到外边缘框
参数中英对照：
- expand: 扩展像素数
- incremental_expandrate: 扩展像素增量
- tapered_corners: 边缘是否倒角
- flip_input: 是否反转输入的蒙版
- blur_radius: 扩展的高斯模糊半径
- lerp_alpha: 蒙版插值因子
- decay_factor: 蒙版衰减因子
- fill_holes: 是否填充内部空洞
"""


    def expand_mask(self, mask, expand, tapered_corners, flip_input, blur_radius, incremental_expandrate, lerp_alpha, decay_factor, fill_holes=False):
        # 保留原始遮罩的副本作为 mask1
        mask1 = mask.clone()

        alpha = lerp_alpha
        decay = decay_factor
        if flip_input:
            mask = 1.0 - mask
        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c],
                           [1, 1, 1],
                           [c, 1, c]])
        growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
        out = []
        previous_output = None
        current_expand = expand
        for m in growmask:
            output = m.numpy().astype(np.float32)
            for _ in range(abs(round(current_expand))):
                if current_expand < 0:
                    output = scipy.ndimage.grey_erosion(output, footprint=kernel)
                else:
                    output = scipy.ndimage.grey_dilation(output, footprint=kernel)
            if current_expand < 0:
                current_expand -= abs(incremental_expandrate)
            else:
                current_expand += abs(incremental_expandrate)
            if fill_holes:
                binary_mask = output > 0
                output = scipy.ndimage.binary_fill_holes(binary_mask)
                output = output.astype(np.float32) * 255
            output = torch.from_numpy(output)
            if alpha < 1.0 and previous_output is not None:
                # 在前一帧和当前帧之间进行插值
                output = alpha * output + (1 - alpha) * previous_output
            if decay < 1.0 and previous_output is not None:
                # 将先前的衰减输出添加到当前帧
                output += decay * previous_output
                output = output / output.max()
            previous_output = output
            out.append(output)

        if blur_radius != 0:
            # 将张量列表转换为PIL图像，应用模糊，然后转换回来
            for idx, tensor in enumerate(out):
                # 将张量转换为PIL图像
                pil_image = tensor2pil(tensor.cpu().detach())[0]
                # 应用高斯模糊
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur_radius))
                # 转换回张量
                out[idx] = pil2tensor(pil_image)
            mask2 = torch.cat(out, dim=0)
        else:
            mask2 = torch.stack(out, dim=0)

        # 计算扩展后的蒙版与原蒙版的差值 mask = mask2 - mask1
        mask = subtract_masks(mask2, mask1)

        return (mask,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "自定义蒙版外框": maskframecafe
}