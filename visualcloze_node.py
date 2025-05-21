# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import torch
import os

import folder_paths
from .models.util import  load_flow_model,load_ae_wrapper
from .visualcloze_wrapper import VisualClozeModel
from .model_utils import gc_cleanup,phi2narry,pre_x_noise_clip_grid,pre_img_grid,nomarl_upscale_tensor,tensortopil_list_upscale,get_sampler_item,tensortopil_list,load_images_list
from .examples.gradio_tasks import dense_prediction_text,conditional_generation_text
from .examples.gradio_tasks_editing import editing_text
from .examples.gradio_tasks_editing_subject import editing_with_subject_text
from .examples.gradio_tasks_photodoodle import photodoodle_text
from .examples.gradio_tasks_relighting import relighting_text
from .examples.gradio_tasks_restoration import image_restoration_text
from .examples.gradio_tasks_style import style_condition_fusion_text,style_transfer_text
from .examples.gradio_tasks_subject import (subject_driven_text,image_restoration_with_subject_text,
                                            style_transfer_with_subject_text,condition_subject_fusion_text,condition_subject_style_fusion_text)
from .examples.gradio_tasks_tryon import tryon_text
from .examples.gradio_tasks_unseen import unseen_tasks_text

MAX_SEED = np.iinfo(np.int32).max
cur_node_path = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

infer_mode_all=dense_prediction_text+conditional_generation_text+editing_text+editing_with_subject_text+photodoodle_text+relighting_text+image_restoration_text+style_condition_fusion_text+style_transfer_text+subject_driven_text+image_restoration_with_subject_text+style_transfer_with_subject_text+condition_subject_fusion_text+condition_subject_style_fusion_text+tryon_text+unseen_tasks_text

class VisualCloze_Aplly:
    def __init__(self):
        self.counters = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt": (["none"] + folder_paths.get_filename_list("checkpoints")+folder_paths.get_filename_list("diffusion_models"),), 
                "lora":(["none"] + folder_paths.get_filename_list("loras"),), 
                "offload": ("BOOLEAN", {"default": True},), }
        }
    
    RETURN_TYPES = ("VisualCloze_PIPE","VisualCloze_INFO")
    RETURN_NAMES = ("model","info",)
    FUNCTION = "main"
    CATEGORY = "VisualCloze"
    
    def main(self, ckpt,lora,offload,):
        print("Loading checkpoint...")
        if ckpt!="none":
            ckpt_path = folder_paths.get_full_path("diffusion_models", ckpt)
        else:
            raise "ckpt is none"
        model_name="flux-dev-fill-lora"

        
        # Load lora model weights
        use_lora=True if lora!="none" and "lora" in model_name  else False


        flow_model = load_flow_model(model_name, ckpt_path,use_lora,offload,device="cpu" if offload else "cuda", lora_rank=256)
       
        resolution=384
        if use_lora:
            lora_path = folder_paths.get_full_path("loras", lora)
            resolution=512 if "512" in lora_path else 384
            print(f"Loading lora model from {lora_path}")
            ckpt = torch.load(lora_path,weights_only=False,map_location='cpu')
            flow_model.load_state_dict(ckpt, strict=False, assign=True)
            del ckpt
            gc_cleanup()
        
       
        pipe=VisualClozeModel(flow_model,device)
        print("Loading checkpoint is done!")
        return (pipe,{"resolution":resolution,"model_name":model_name},)


class VisualCloze_CLIPText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae":("VAE",),
                "clip": ("CLIP",),
                "info": ("VisualCloze_INFO",),
                "query_image":("IMAGE",),
                "in_context_img_1":("IMAGE",), #始终开启上下文学习
                "infer_tasks": (["none"]+infer_mode_all,),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED,}),
                # "width": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 16, "display": "number"}),
                # "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 16, "display": "number"}),
                "content_prompt":("STRING", {"multiline": True,"default": ""}),
            },
            "optional": {
                        "in_context_img_2":("IMAGE",),
                        "in_context_img_3":("IMAGE",),
                        "in_context_img_4":("IMAGE",),
            }
        }

    RETURN_TYPES = ("VisualCloze_EMB",)
    RETURN_NAMES = ("emb_dict",)
    FUNCTION = "encode"
    CATEGORY = "VisualCloze"


    def encode(self, vae,clip,info, query_image,in_context_img_1,infer_tasks,seed, content_prompt,**kwargs):
        # use normal vae 
        vae=load_ae_wrapper(info.get("model_name"),vae.get_sd())
        vae.requires_grad_(False)

        assert infer_tasks!="none","please choice a task"

        # pre imge and get grid
        # in_pil_img_1=tensortopil_list_upscale(kwargs.get("in_context_img_1"),width,height) if isinstance(kwargs.get("in_context_img_1"),torch.Tensor) else None
        # in_pil_img_2=tensortopil_list_upscale(kwargs.get("in_context_img_2"),width,height) if isinstance(kwargs.get("in_context_img_2"),torch.Tensor) else None
        # in_pil_img_3=tensortopil_list_upscale(kwargs.get("in_context_img_3"),width,height)  if isinstance(kwargs.get("in_context_img_3"),torch.Tensor) else None

        in_pil_img_2=tensortopil_list(kwargs.get("in_context_img_2")) if isinstance(kwargs.get("in_context_img_2"),torch.Tensor) else None
        in_pil_img_3=tensortopil_list(kwargs.get("in_context_img_3")) if isinstance(kwargs.get("in_context_img_3"),torch.Tensor) else None
        in_pil_img_4=tensortopil_list(kwargs.get("in_context_img_4"))  if isinstance(kwargs.get("in_context_img_4"),torch.Tensor) else None

        grid_imag_list,grid_w,grid_h=pre_img_grid(query_image, in_context_img_1, in_pil_img_2, in_pil_img_3, in_pil_img_4)

        print(grid_imag_list,grid_w,grid_h)

        outputs=get_sampler_item(infer_tasks,grid_w,grid_h,content_prompt)
    
        generator = torch.Generator(device=device).manual_seed(int(seed))
        # clip
        inp,img_cond,sliced_subimage,mask_position,upsampling_size = pre_x_noise_clip_grid(clip,vae,generator,grid_imag_list,
                                                                                           [outputs[1]+" "+ outputs[2]+" "+outputs[3]], grid_h,grid_w,info.get("resolution"),device,dtype=torch.bfloat16)
             
        emb_dict={"sliced_subimage":sliced_subimage, "mask_position":mask_position,"inp":inp,"img_cond":img_cond,"clip":clip,"grid_h":grid_h,"grid_w":grid_w,
                  "content_prompt":outputs[3],"generator":generator,"vae":vae,"upsampling_size":upsampling_size,"upsampling_noise":outputs[4],"steps":outputs[5]}
        return (emb_dict,)
    
class VisualCloze_KSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("VisualCloze_PIPE",),
                "emb_dict": ("VisualCloze_EMB",),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000,}),
                "upsampling_steps": ("INT", {"default": 10, "min": 1, "max": 10000,}),
                "upsampling_noise": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1,}),
                "cfg": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 100.0, "step": 0.1,}),
            },
                
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    CATEGORY = "VisualCloze"


    def sample(self, model,emb_dict, steps,upsampling_steps,upsampling_noise, cfg,):

        result = model.process_images(
                emb_dict.get("vae"), 
                emb_dict.get("clip"), 
                cfg,
                steps=steps if emb_dict.get("steps") is None else emb_dict.get("steps"),
                upsampling_steps=upsampling_steps,
                upsampling_noise=upsampling_noise if emb_dict.get("upsampling_noise") is None else emb_dict.get("upsampling_noise"),
                inp=emb_dict.get("inp"),
                img_cond=emb_dict.get("img_cond"), 
                sliced_subimage=emb_dict.get("sliced_subimage"),
                mask_position=emb_dict.get("mask_position"),
                upsampling_size=emb_dict.get("upsampling_size"),
                content_prompt=emb_dict.get("content_prompt"),
                generator=emb_dict.get("generator"),
                grid_h=emb_dict.get("grid_h"),
                grid_w=emb_dict.get("grid_w"),
                )[-1]
        return (phi2narry(result),)


class Img_Quadruple:
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image_1": ("IMAGE",), # [B,H,W,C], C=3
                             },
                "optional": {"image_2": ("IMAGE",),
                             "image_3": ("IMAGE",),
                             "image_4": ("IMAGE",)}
                }
    
    RETURN_TYPES = ("IMAGE",)
    ETURN_NAMES = ("image",)
    FUNCTION = "main"
    CATEGORY = "VisualCloze"
    
    def main(self, image_1, **kwargs):

        B,height,width, _ = image_1.size()
        assert B == 1 , "only support batch size 1"

        image_2 = nomarl_upscale_tensor(kwargs.get("image_2"), width, height) if isinstance(kwargs.get("image_2"), torch.Tensor) else None
        image_3 = nomarl_upscale_tensor(kwargs.get("image_3"), width, height) if isinstance(kwargs.get("image_3"), torch.Tensor) else None
        image_4 = nomarl_upscale_tensor(kwargs.get("image_4"), width, height) if isinstance(kwargs.get("image_4"), torch.Tensor) else None

        img_list = [image_1]
        for img in [image_2, image_3, image_4]:
            if img is not None:
                C,_,_, _ = img.size()
                assert C == 1 , "only support batch size 1"
                img_list.append(img)

        images = torch.cat(tuple(img_list), dim=0)
        return (images,)

NODE_CLASS_MAPPINGS = {
    "VisualCloze_Aplly": VisualCloze_Aplly,
    "VisualCloze_KSampler": VisualCloze_KSampler,
    "VisualCloze_CLIPText": VisualCloze_CLIPText,
    "Img_Quadruple": Img_Quadruple,
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MSdiffusion_Aplly": "MSdiffusion_Aplly",
    "VisualCloze_KSampler": "VisualCloze_KSampler",
    "VisualCloze_CLIPText": "VisualCloze_CLIPText",
    "Img_Quadruple": "Img_Quadruple",
}
