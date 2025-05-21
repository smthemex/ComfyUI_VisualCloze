# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
from PIL import Image
import numpy as np
import cv2
from einops import rearrange
from .models.sampling import prepare_modified_wrapper
from torchvision import transforms
import torch.nn.functional as F
from .util.imgproc import to_rgb_if_rgba

from comfy.utils import common_upscale,ProgressBar


cur_path = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

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



image_transform = transforms.Compose([
            transforms.Lambda(lambda img: to_rgb_if_rgba(img)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

SAMPLER_NAMES = ["euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2","dpm_2", "dpm_2_ancestral",
                  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
                  "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",
                  "ipndm", "ipndm_v", "deis","ddim", "uni_pc", "uni_pc_bh2"]

SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta"]



def get_sampler_item(infer_tasks,grid_w,grid_h,content_prompt):
    if infer_tasks in dense_prediction_text:
        from .examples.gradio_tasks import process_dense_prediction_tasks_w
        outputs=process_dense_prediction_tasks_w(infer_tasks,grid_w,grid_h) #mask, layout_prompt, task_prompt, content_prompt, upsampling_noise, steps
    elif infer_tasks in conditional_generation_text:
        from.examples.gradio_tasks import process_conditional_generation_tasks_w
        outputs=process_conditional_generation_tasks_w(infer_tasks,grid_w,grid_h,content_prompt) # TODO condition 的前后景需要应用mask遮罩
    elif infer_tasks in editing_text:
        from.examples.gradio_tasks_editing import process_editing_tasks_w
        outputs=process_editing_tasks_w(infer_tasks,grid_w,grid_h,content_prompt)
    elif infer_tasks in editing_with_subject_text:
        from.examples.gradio_tasks_editing_subject import process_editing_with_subject_tasks_w
        outputs=process_editing_with_subject_tasks_w(infer_tasks,grid_w,grid_h,content_prompt)
    elif infer_tasks in photodoodle_text:
        from.examples.gradio_tasks_photodoodle import process_photodoodle_tasks_w
        outputs=process_photodoodle_tasks_w(infer_tasks,grid_w,grid_h,content_prompt)
    elif infer_tasks in relighting_text:
        from.examples.gradio_tasks_relighting import process_relighting_tasks_w
        outputs=process_relighting_tasks_w(infer_tasks,grid_w,grid_h,content_prompt)
    elif infer_tasks in image_restoration_text:
        from.examples.gradio_tasks_restoration import process_image_restoration_tasks_w #TODO
        outputs=process_image_restoration_tasks_w(infer_tasks,grid_w,grid_h,content_prompt)
    elif infer_tasks in style_condition_fusion_text:
        from.examples.gradio_tasks_style import process_style_condition_fusion_tasks_w #TODO
        outputs=process_style_condition_fusion_tasks_w(infer_tasks,grid_w,grid_h)
    elif infer_tasks in style_transfer_text:
        from.examples.gradio_tasks_style import process_style_transfer_tasks_w #TODO
        outputs=process_style_transfer_tasks_w(infer_tasks,grid_w,grid_h)
    elif infer_tasks in subject_driven_text:
        from.examples.gradio_tasks_subject import process_subject_driven_tasks_w #TODO
        outputs=process_subject_driven_tasks_w(infer_tasks,grid_w,grid_h,content_prompt)
    elif infer_tasks in image_restoration_with_subject_text:
        from.examples.gradio_tasks_subject import process_image_restoration_with_subject_tasks_w #TODO
        outputs=process_image_restoration_with_subject_tasks_w(infer_tasks,grid_w,grid_h,content_prompt)
    elif infer_tasks in style_transfer_with_subject_text:
        from.examples.gradio_tasks_subject import process_style_transfer_with_subject_tasks_w #TODO
        outputs=process_style_transfer_with_subject_tasks_w(infer_tasks,grid_w,grid_h)
    elif infer_tasks in condition_subject_style_fusion_text:
        from.examples.gradio_tasks_subject import process_condition_subject_style_fusion_tasks_w #TODO
        outputs=process_condition_subject_style_fusion_tasks_w(infer_tasks,grid_w,grid_h)
    elif infer_tasks in condition_subject_fusion_text:
        from.examples.gradio_tasks_subject import process_condition_subject_fusion_tasks_w #TODO
        outputs=process_condition_subject_fusion_tasks_w(infer_tasks,grid_w,grid_h,content_prompt)
    elif infer_tasks in tryon_text:
        from.examples.gradio_tasks_tryon import process_tryon_tasks_w #TODO
        outputs=process_tryon_tasks_w(infer_tasks,grid_w,grid_h)
    elif infer_tasks in unseen_tasks_text:
        from.examples.gradio_tasks_unseen import process_unseen_tasks_w #TODO
        outputs=process_unseen_tasks_w(infer_tasks,grid_w,grid_h)
    print(outputs)

    return outputs




def convert_cfvae2diffuser(VAE):
    from diffusers import AutoencoderKL
   
    vae_config = os.path.join(cur_path, "configs/FLUX.1-dev/vae/config.json")
    vae_state_dict=VAE.get_sd()
    ae_config = AutoencoderKL.load_config(vae_config)
    AE = AutoencoderKL.from_config(ae_config).to(device, torch.bfloat16)
    AE.load_state_dict(vae_state_dict, strict=False)
    del vae_state_dict
    gc_cleanup()
    return AE


def resize_with_aspect_ratio(img, resolution, divisible=16, aspect_ratio=None):
    """Resize image while maintaining aspect ratio, ensuring area is close to resolution**2 and dimensions are divisible by 16
    
    Args:
        img: PIL Image or torch.Tensor (C,H,W)/(B,C,H,W)
        resolution: target resolution
        divisible: ensure output dimensions are divisible by this number
    
    Returns:
        Resized image of the same type as input
    """
    # Check input type and get dimensions
    is_tensor = isinstance(img, torch.Tensor)
    if is_tensor:
        if img.dim() == 3:
            c, h, w = img.shape
            batch_dim = False
        else:
            b, c, h, w = img.shape
            batch_dim = True
    else:
        w, h = img.size
        
    # Calculate new dimensions
    if aspect_ratio is None:
        aspect_ratio = w / h
    target_area = resolution * resolution
    new_h = int((target_area / aspect_ratio) ** 0.5)
    new_w = int(new_h * aspect_ratio)
    
    # Ensure divisible by divisible
    new_w = max(new_w // divisible, 1) * divisible
    new_h = max(new_h // divisible, 1) * divisible
    
    # Adjust size based on input type
    if is_tensor:
        # Use torch interpolation method
        mode = 'bilinear'
        align_corners = False
        if batch_dim:
            return F.interpolate(img, size=(new_h, new_w), 
                               mode=mode, align_corners=align_corners)
        else:
            return F.interpolate(img.unsqueeze(0), size=(new_h, new_w),
                               mode=mode, align_corners=align_corners).squeeze(0)
    else:
        # Use PIL LANCZOS resampling
        return img.resize((new_w, new_h), Image.LANCZOS)

def pre_img_grid(query_image, in_context_img, in_pil_img_1, in_pil_img_2, in_pil_img_3): 
    """
    list[[in_pil_img],...,[q_pil_image]]
    """

    # q_pil_image=tensortopil_list_upscale(query_image,width, height)
    # in_pil_img=tensortopil_list_upscale(in_context_img,width, height) 

    q_pil_image=tensortopil_list(query_image)
    in_pil_img=tensortopil_list(in_context_img) 
    
    grid_w=len(in_pil_img)

    assert grid_w>=2,"in_context_img must be a list of images, and the length of the list must be greater than 2"
    assert grid_w==len(q_pil_image)+1,"query_image must be less 1 than in_context_img length"

    
    for img in [in_pil_img_1, in_pil_img_2, in_pil_img_3]:
        if img is not None:
           assert len(img)==grid_w,"in_context_img must be a list of images, and the length of the list must be equal to in_pil_img length"

    # 计数非 None 的数量
    grid_h = sum(1 for img in [in_pil_img_1, in_pil_img_2, in_pil_img_3] if img is not None)+2 #2是包含Query_image和in_context_img


    # 规范网格图片的数量
    q_pil_image=q_pil_image+[None]
    grid_imag_list=[in_pil_img, in_pil_img_1, in_pil_img_2, in_pil_img_3, q_pil_image] # Query_image在最后的row要改成none
    grid_imag_list = [img for img in grid_imag_list if img is not None]
    return grid_imag_list,grid_w,grid_h
    


def pre_x_noise_clip(clip,ae,generator,image,prompt,upsampling_noise,device,dtype,target_size):

    if target_size is None:
        aspect_ratio = 1
        target_area = 1024 * 1024
        new_h = int((target_area / aspect_ratio) ** 0.5)
        new_w = int(new_h * aspect_ratio)
        target_size = (new_w, new_h)

    if target_size[0] * target_size[1] > 1024 * 1024:
        aspect_ratio = target_size[0] / target_size[1]
        target_area = 1024 * 1024
        new_h = int((target_area / aspect_ratio) ** 0.5)
        new_w = int(new_h * aspect_ratio)
        target_size = (new_w, new_h)
    
    image = image.resize(((target_size[0] // 16) * 16, (target_size[1] // 16) * 16))

    processed_image = image_transform(image)
    processed_image = processed_image.to(device, non_blocking=True)
    blank = torch.zeros_like(processed_image, device=device, dtype=dtype)
    mask = torch.full((1, 1, processed_image.shape[1], processed_image.shape[2]), fill_value=1, device=device, dtype=dtype)
    with torch.no_grad():
        # latent = ae.encode(processed_image[None].to(dtype)).latent_dist.sample()
        # blank = ae.encode(blank[None].to(dtype)).latent_dist.sample()
        latent = ae.encode(processed_image[None].to(dtype))
        blank = ae.encode(blank[None].to(dtype))
        # latent = (latent - ae.config.shift_factor) * ae.config.scaling_factor
        # blank = (blank - ae.config.shift_factor) * ae.config.scaling_factor
        latent = (latent - ae.shift_factor) * ae.scale_factor
        blank = (blank - ae.shift_factor) * ae.scale_factor
        latent_h, latent_w = latent.shape[2:]

        mask = rearrange(mask, "b c (h ph) (w pw) -> b (c ph pw) h w", ph=8, pw=8) 
        mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        
        latent = latent.to(dtype)
        blank = blank.to(dtype)
        latent = rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        blank = rearrange(blank, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            
        img_cond = torch.cat((blank, mask), dim=-1)
        noise = torch.randn([1, 16, latent_h, latent_w], device=device, generator=generator).to(dtype)
        x = [[noise]]
        
        inp = prepare_modified_wrapper(clip, img=x, prompt=[prompt], proportion_empty_prompts=0.0)
        inp["img"] = inp["img"] * (1 - upsampling_noise) + latent * upsampling_noise
    return inp,img_cond.to(device,torch.bfloat16),latent_h, latent_w


def pre_x_noise_clip_grid(clip,ae,generator,images,prompt,grid_h,grid_w,resolution,device,dtype):
    # Ensure all images are RGB mode or None
    # for i in range(0, grid_h): #2行*2列
    #     images[i] = [img.convert("RGB") if img is not None else None for img in images[i]]
    processed_images = []
    mask_position = []
    target_size = None
    upsampling_size = None
    
    for i in range(grid_h):
        # Find the size of the first non-empty image in this row
        reference_size = None
        for j in range(0, grid_w):
            if images[i][j] is not None:
                if i == grid_h - 1 and upsampling_size is None:
                    upsampling_size = images[i][j].size

                resized = resize_with_aspect_ratio(images[i][j], resolution, aspect_ratio=None)
                reference_size = resized.size
                if i == grid_h - 1 and target_size is None:
                    target_size = reference_size
                break
        
        # Process all images in this row
        for j in range(0, grid_w):
            if images[i][j] is not None:
                target = resize_with_aspect_ratio(images[i][j], resolution, aspect_ratio=None)
                if target.width <= target.height:
                    target = target.resize((reference_size[0], int(reference_size[0] / target.width * target.height)))
                    target = center_crop(target, reference_size)
                elif target.width > target.height:
                    target = target.resize((int(reference_size[1] / target.height * target.width), reference_size[1]))
                    target = center_crop(target, reference_size)
                
                processed_images.append(target)
                if i == grid_h - 1:
                    mask_position.append(0)
            else:
                # If this row has a reference size, use it; otherwise use default size
                if reference_size:
                    blank = Image.new('RGB', reference_size, (0, 0, 0))
                else:
                    blank = Image.new('RGB', (resolution, resolution), (0, 0, 0))
                processed_images.append(blank)
                if i == grid_h - 1:
                    mask_position.append(1)
                else:
                    raise ValueError('Please provide each image in the in-context example.')
        
    # return processed_images
    
    if len(mask_position) > 1 and sum(mask_position) > 1:
        if target_size is None:
            new_w = 384
        else:
            new_w = target_size[0]
        for i in range(len(processed_images)):
            if processed_images[i] is not None:
                new_h = int(processed_images[i].height * (new_w / processed_images[i].width))
                new_w = int(new_w / 16) * 16
                new_h = int(new_h / 16) * 16
                processed_images[i] = processed_images[i].resize((new_w, new_h))
    with torch.autocast("cuda", dtype):            
        grid_image = []
        fill_mask = []
        for i in range(grid_h):
            row_images = [image_transform(img) for img in processed_images[i * grid_w: (i + 1) * grid_w]]
            if i == grid_h - 1:
                row_masks = [torch.full((1, 1, row_images[0].shape[1], row_images[0].shape[2]), fill_value=m, device=device) for m in mask_position]
            else:
                row_masks = [torch.full((1, 1, row_images[0].shape[1], row_images[0].shape[2]), fill_value=0, device=device) for m in mask_position]

            grid_image.append(torch.cat(row_images, dim=2).to(device, non_blocking=True))
            fill_mask.append(torch.cat(row_masks, dim=3))

        # Encode condition image
        with torch.no_grad():
            #fill_cond = [ae.encode(img[None].to(dtype)).latent_dist.sample()[0] for img in grid_image]
            fill_cond = [ae.encode(img[None].to(dtype)) for img in grid_image]#torch.Size([1, 16, 48, 144])
            #print(fill_cond[0].shape)
            #fill_cond = [(img - ae.config.shift_factor) * ae.config.scale_factor for img in fill_cond]
            fill_cond = [(img - ae.shift_factor) * ae.scale_factor for img in fill_cond]
            
            # Rearrange mask
            fill_mask = [rearrange(mask, "b c (h ph) (w pw) -> b (c ph pw) h w", ph=8, pw=8) for mask in fill_mask]
            fill_mask = [rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2) for mask in fill_mask]
        
        fill_cond = [img.to(dtype) for img in fill_cond]
        #fill_cond = [rearrange(img.unsqueeze(0), "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2) for img in fill_cond]
        fill_cond = [rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2) for img in fill_cond]
        
        fill_cond =  torch.cat(fill_cond, dim=1)
        fill_mask =  torch.cat(fill_mask, dim=1)
        img_cond = torch.cat((fill_cond, fill_mask), dim=-1)

        # Generate sample
        noise = []
        sliced_subimage = []
      
        for sub_img in grid_image:
            h, w = sub_img.shape[-2:]
            sliced_subimage.append((h, w))
            latent_w, latent_h = w // 8, h // 8
            noise.append(torch.randn([1, 16, latent_h, latent_w], device=device, generator=generator).to(dtype))
        x = [noise]
        with torch.no_grad():
            inp = prepare_modified_wrapper(clip, img=x, prompt=[prompt], proportion_empty_prompts=0.0)
       
    return inp,img_cond.to(device,torch.bfloat16),sliced_subimage,mask_position,upsampling_size



def gc_cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def center_crop(image, target_size):
    width, height = image.size
    new_width, new_height = target_size

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    return image.crop((left, top, right, bottom))

def phi2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

def tensor_to_image(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def tensortopil_list(tensor_in):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [tensor_to_image(tensor_in)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[tensor_to_image(i) for i in tensor_list]
    return img_list

def tensortopil_list_upscale(tensor_in,width,height):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [nomarl_upscale(tensor_in,width,height)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[nomarl_upscale(i,width,height) for i in tensor_list]
    return img_list

def tensortolist(tensor_in,width,height):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        tensor_list = [nomarl_tensor_upscale(tensor_in,width,height)]
    else:
        tensor_list_ = torch.chunk(tensor_in, chunks=d1)
        tensor_list=[nomarl_tensor_upscale(i,width,height) for i in tensor_list_]
    return tensor_list



def nomarl_tensor_upscale(tensor, width, height):
    samples = tensor.movedim(-1, 1)
    samples = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = samples.movedim(1, -1)
    return samples
def nomarl_upscale(img, width, height):
    samples = img.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img = tensor_to_image(samples)
    return img
def nomarl_upscale_tensor(img, width, height):
    samples = img.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    return samples
    


def images_generator(img_list: list, ):
    # get img size
    sizes = {}
    for image_ in img_list:
        if isinstance(image_, Image.Image):
            count = sizes.get(image_.size, 0)
            sizes[image_.size] = count + 1
        elif isinstance(image_, np.ndarray):
            count = sizes.get(image_.shape[:2][::-1], 0)
            sizes[image_.shape[:2][::-1]] = count + 1
        else:
            raise "unsupport image list,must be pil or cv2!!!"
    size = max(sizes.items(), key=lambda x: x[1])[0]
    yield size[0], size[1]
    
    # any to tensor
    def load_image(img_in):
        if isinstance(img_in, Image.Image):
            img_in = img_in.convert("RGB")
            i = np.array(img_in, dtype=np.float32)
            i = torch.from_numpy(i).div_(255)
            if i.shape[0] != size[1] or i.shape[1] != size[0]:
                i = torch.from_numpy(i).movedim(-1, 0).unsqueeze(0)
                i = common_upscale(i, size[0], size[1], "lanczos", "center")
                i = i.squeeze(0).movedim(0, -1).numpy()
            return i
        elif isinstance(img_in, np.ndarray):
            i = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB).astype(np.float32)
            i = torch.from_numpy(i).div_(255)
            print(i.shape)
            return i
        else:
            raise "unsupport image list,must be pil,cv2 or tensor!!!"
    
    total_images = len(img_list)
    processed_images = 0
    pbar = ProgressBar(total_images)
    images = map(load_image, img_list)
    try:
        prev_image = next(images)
        while True:
            next_image = next(images)
            yield prev_image
            processed_images += 1
            pbar.update_absolute(processed_images, total_images)
            prev_image = next_image
    except StopIteration:
        pass
    if prev_image is not None:
        yield prev_image


def load_images_list(img_list: list, ):
    gen = images_generator(img_list)
    (width, height) = next(gen)
    images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3)))))
    if len(images) == 0:
        raise FileNotFoundError(f"No images could be loaded .")
    return images
    



