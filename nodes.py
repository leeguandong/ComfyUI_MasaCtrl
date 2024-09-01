import os
import torch
import torch.nn.functional as F
import numpy as np

import folder_paths
import comfy.model_management as mm

from PIL import Image
from diffusers import DDIMScheduler
from huggingface_hub import snapshot_download, hf_hub_download
from torchvision.utils import save_image
from pytorch_lightning import seed_everything
from torchvision.io import read_image

from .masactrl.diffuser_utils import MasaCtrlPipeline
from .masactrl.masactrl_utils import AttentionBase, regiter_attention_editor_diffusers
from .masactrl.masactrl import MutualSelfAttentionControl, MutualSelfAttentionControlMask, \
    MutualSelfAttentionControlMaskAuto


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def convert_preview_image(images):
    # 转换图像为 torch.Tensor，并调整维度顺序为 NHWC
    images_tensors = []
    for img in images:
        # 将 PIL.Image 转换为 numpy.ndarray
        img_array = np.array(img)
        # 转换 numpy.ndarray 为 torch.Tensor
        img_tensor = torch.from_numpy(img_array).float() / 255.
        # 转换图像格式为 CHW (如果需要)
        if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
            img_tensor = img_tensor.permute(2, 0, 1)
        # 添加批次维度并转换为 NHWC
        img_tensor = img_tensor.unsqueeze(0).permute(0, 2, 3, 1)
        images_tensors.append(img_tensor)

    if len(images_tensors) > 1:
        output_image = torch.cat(images_tensors, dim=0)
    else:
        output_image = images_tensors[0]
    return output_image


class MasaCtrlModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    ["runwayml/stable-diffusion-v1-5","Yntec/stable-diffusion-v1-5"],
                    {"default": "runwayml/stable-diffusion-v1-5"},
                ),
                "scheduler": ("SCHEDULER",)
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "MasaCtrl"

    def load_checkpoint(self, model, scheduler):
        device = mm.get_torch_device()

        model = MasaCtrlPipeline.from_pretrained(model, scheduler=scheduler).to(device)
        return (model,)


class MasaCtrlLoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"
    CATEGORY = "MasaCtrl"

    def load_image(self, image):
        device = mm.get_torch_device()
        image_path = folder_paths.get_annotated_filepath(image)

        image = read_image(image_path)
        image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
        image = F.interpolate(image, (512, 512))
        image = image.to(device)
        return (image,)


class MasaCtrlInversion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "source_prompt": ("STRING", {"forceInput": True, "default": ""}),
                "target_prompt": ("STRING", {"forceInput": True, "default": ""}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "guidance_scale": (
                    "FLOAT", {"default": 7.5, "min": 0.0, "max": 10.0, "step": 0.5, "display": "slider"}),
                "return_intermediates": ("BOOLEAN", {"default": True, }),
            }
        }

    RETURN_TYPES = ("LATENTS",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "inversion"
    CATEGORY = "MasaCtrl"

    def inversion(self,
                  image,
                  model,
                  source_prompt,
                  target_prompt,
                  num_inference_steps,
                  guidance_scale,
                  return_intermediates):
        device = mm.get_torch_device()

        prompts = [source_prompt, target_prompt]
        start_code, latents_list = model.invert(image,
                                                source_prompt,
                                                guidance_scale=guidance_scale,
                                                num_inference_steps=num_inference_steps,
                                                return_intermediates=return_intermediates)
        start_code = start_code.expand(len(prompts), -1, -1, -1)
        return (start_code,)


class MasaCtrlConcatImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "image_masactrl": ("IMAGE",),
                "image_fixed": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "concat"
    CATEGORY = "MasaCtrl"

    def concat(self, source_image, image_masactrl, image_fixed):
        source_image = source_image * 0.5 + 0.5
        image_masactrl1 = image_masactrl[0:1]
        image_masactrl2 = image_masactrl[-1:]

        images = torch.cat([source_image, image_masactrl1, image_fixed, image_masactrl2], dim=0)
        # import pdb;pdb.set_trace()
        image_ = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (image_ * 255).astype(np.uint8)
        pil_images = [Image.fromarray(image) for image in images]
        # pil_images = [Image.fromarray(image) for image in images.cpu()]
        # pil_images = [Image.fromarray(image_tensor.cpu().numpy().transpose(1, 2, 0)) for image_tensor in images]
        output_images = convert_preview_image(pil_images)
        return (output_images,)


class DirectSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latents": ("LATENTS",),
                "target_prompt": ("STRING", {"forceInput": True, "default": ""}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "guidance_scale": (
                    "FLOAT", {"default": 7.5, "min": 0.0, "max": 10.0, "step": 0.5, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "direct_sampler"
    CATEGORY = "MasaCtrl"

    def direct_sampler(self,
                       model,
                       latents,
                       target_prompt,
                       num_inference_steps,
                       guidance_scale):
        # device = mm.get_torch_device()

        editor = AttentionBase()
        regiter_attention_editor_diffusers(model, editor)
        image_fixed = model([target_prompt],
                            latents=latents[-1:],
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale)

        return (image_fixed,)


class MutualSelfAttentionControlSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latents": ("LATENTS",),
                "source_prompt": ("STRING", {"forceInput": True, "default": ""}),
                "target_prompt": ("STRING", {"forceInput": True, "default": ""}),
                "step": ("INT", {"default": 4, "min": 1, "max": 20}),
                "layer": ("INT", {"default": 10, "min": 1, "max": 100}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "guidance_scale": (
                    "FLOAT", {"default": 7.5, "min": 0.0, "max": 10.0, "step": 0.5, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "mutual_self_attention_control"
    CATEGORY = "MasaCtrl"

    def mutual_self_attention_control(self,
                                      model,
                                      latents,
                                      source_prompt,
                                      target_prompt,
                                      step,
                                      layer,
                                      num_inference_steps,
                                      guidance_scale):
        prompts = [source_prompt, target_prompt]
        editor = MutualSelfAttentionControl(step, layer)
        regiter_attention_editor_diffusers(model, editor)
        image_masactrl = model(prompts,
                               latents=latents,
                               guidance_scale=guidance_scale)
        return (image_masactrl,)


class MutualSelfAttentionControlMaskAutoSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latents": ("LATENTS",),
                "source_prompt": ("STRING", {"forceInput": True, "default": ""}),
                "target_prompt": ("STRING", {"forceInput": True, "default": ""}),
                "step": ("INT", {"default": 4, "min": 1, "max": 20}),
                "layer": ("INT", {"default": 10, "min": 1, "max": 100}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "guidance_scale": (
                    "FLOAT", {"default": 7.5, "min": 0.0, "max": 10.0, "step": 0.5, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "mutual_self_attention_control_mask"
    CATEGORY = "MasaCtrl"

    def mutual_self_attention_control_mask(self,
                                           model,
                                           latents,
                                           source_prompt,
                                           target_prompt,
                                           step,
                                           layer,
                                           num_inference_steps,
                                           guidance_scale):
        prompts = [source_prompt, target_prompt]
        editor = MutualSelfAttentionControlMaskAuto(step, layer)
        regiter_attention_editor_diffusers(model, editor)
        image_masactrl = model(prompts,
                               latents=latents,
                               guidance_scale=guidance_scale)
        return (image_masactrl,)


NODE_CLASS_MAPPINGS = {
    "MasaCtrlLoadImage": MasaCtrlLoadImage,
    "MasaCtrlModelLoader": MasaCtrlModelLoader,
    "MasaCtrlInversion": MasaCtrlInversion,
    "MasaCtrlConcatImage": MasaCtrlConcatImage,
    "DirectSampler": DirectSampler,
    "MutualSelfAttentionControlSampler": MutualSelfAttentionControlSampler,
    "MutualSelfAttentionControlMaskAutoSampler": MutualSelfAttentionControlMaskAutoSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MasaCtrlLoadImage": "MasaCtrl LoadImage",
    "MasaCtrlModelLoader": "MasaCtrl Model Loader",
    "MasaCtrlInversion": "MasaCtrl Inversion",
    "MasaCtrlConcatImage": "MasaCtrl Concat Image",
    "DirectSampler": "Direct Sampler",
    "MutualSelfAttentionControlSampler": "MutualSelfAttentionControl Sampler",
    "MutualSelfAttentionControlMaskSampler": "MutualSelfAttentionControlMaskAuto Sampler"
}
