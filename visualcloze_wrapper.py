

from einops import rearrange
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from .transport import Sampler, create_transport

from .util.imgproc import to_rgb_if_rgba
from .model_utils import pre_x_noise_clip

content_instruction = [
            "The content of the last image in the final row is: ",
            "The last image of the last row depicts: ",
            "In the final row, the last image shows: ",
            "The last image in the bottom row illustrates: ",
            "The content of the bottom-right image is: ",
            "The final image in the last row portrays: ",
            "The last image of the final row displays: ",
            "In the last row, the final image captures: ",
            "The bottom-right corner image presents: ",
            "The content of the last image in the concluding row is: ",
            "In the last row, ",
            "The editing instruction in the last row is: ", 
        ]






class VisualClozeModel:
    def __init__(
        self,
        flow_model,
        device, 
        atol=1e-6,
        rtol=1e-3, 
        solver='euler', 
        time_shifting_factor=1, 
        precision='bf16'
        ):
        self.atol = atol
        self.rtol = rtol
        self.solver = solver
        self.time_shifting_factor = time_shifting_factor
        # self.resolution = resolution
        self.precision = precision
        #self.max_length = max_length
        #self.lora_rank = lora_rank
        self.device = device
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.precision]
        self.model=flow_model
        # Initialize model
        # print("Initializing model...")
        # self.model = load_flow_model(model_name, device=self.device, lora_rank=self.lora_rank)
        
        # Initialize VAE
       # print("Initializing VAE...")
        # self.ae = AutoencoderKL.from_pretrained(f"black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=self.dtype).to(self.device)
        # self.ae.requires_grad_(False)
        
        # Initialize text encoders
        #print("Initializing text encoders...")
        # self.t5 = load_t5(self.device, max_length=self.max_length)
        # self.clip = load_clip(self.device)
        
        self.model.eval().to(self.device)
        
        # Initialize sampler
        transport = create_transport(
            "Linear",
            "velocity",
            do_shift=True,
        ) 
        self.sampler = Sampler(transport)
        # self.sample_fn = self.sampler.sample_ode(
        #     sampling_method=self.solver,
        #     num_steps=30,
        #     atol=self.atol,
        #     rtol=self.rtol,
        #     reverse=False,
        #     do_shift=True,
        #     time_shifting_factor=self.time_shifting_factor,
        # )
        
        # Image transformation
        # self.image_transform = transforms.Compose([
        #     transforms.Lambda(lambda img: to_rgb_if_rgba(img)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        # ])
        
        # self.grid_h = None
        # self.grid_w = None
        
    # def set_grid_size(self, h, w):
    #     """Set grid size"""
    #     self.grid_h = h
    #     self.grid_w = w
    
    def upsampling(self, ae,image, target_size, cfg, upsampling_steps, upsampling_noise,inp,img_cond,latent_h, latent_w,):
        # content_instruction = [
        #     "The content of the last image in the final row is: ",
        #     "The last image of the last row depicts: ",
        #     "In the final row, the last image shows: ",
        #     "The last image in the bottom row illustrates: ",
        #     "The content of the bottom-right image is: ",
        #     "The final image in the last row portrays: ",
        #     "The last image of the final row displays: ",
        #     "In the last row, the final image captures: ",
        #     "The bottom-right corner image presents: ",
        #     "The content of the last image in the concluding row is: ",
        #     "In the last row, ",
        #     "The editing instruction in the last row is: ", 
        # ]
        # for c in content_instruction:
        #     if content_prompt.startswith(c):
        #         content_prompt = content_prompt.replace(c, '')
        
        
        if upsampling_noise >= 1.0:
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
            return image

        self.sample_fn = self.sampler.sample_ode(
            sampling_method=self.solver,
            num_steps=upsampling_steps,
            atol=self.atol,
            rtol=self.rtol,
            reverse=False,
            do_shift=False,
            time_shifting_factor=1.0, 
            strength=upsampling_noise
        )

        with torch.no_grad():

            model_kwargs = dict(
                txt=inp["txt"], 
                txt_ids=inp["txt_ids"], 
                txt_mask=inp["txt_mask"],
                y=inp["vec"], 
                img_ids=inp["img_ids"], 
                img_mask=inp["img_mask"], 
                cond=img_cond,
                guidance=torch.full((1,), cfg, device=self.device, dtype=self.dtype),
            )
            sample = self.sample_fn(inp["img"], self.model.forward, model_kwargs)[-1]
            
            sample = sample[:1]
            sample = rearrange(sample.float(), "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=2, pw=2, h=latent_h // 2, w=latent_w // 2)
            print("sample", sample.shape) 
            sample = ae.decode(sample / ae.scale_factor + ae.shift_factor) #ae.decode(sample / ae.config.scaling_factor + ae.config.shift_factor)[0]
            #print("sample", sample.shape) #sample torch.Size([1, 3, 768, 512])
            sample = (sample + 1.0) / 2.0
            sample.clamp_(0.0, 1.0)
            sample = sample[0]
            
            output_image = to_pil_image(sample.float())
            
            return output_image
    
    def process_images( 
        self,
        ae,
        clip,
        # images: list[list[Image.Image]], 
        # prompts: list[str], 
        #seed: int = 0, 
        cfg: int = 30, 
        steps: int = 30, 
        upsampling_steps: int = 10, 
        upsampling_noise: float = 0.4, 
        is_upsampling: bool =True,
        inp: dict = None,
        img_cond= None,
        sliced_subimage= None,
        mask_position= None,
        upsampling_size= None,
        content_prompt: str = "",
        generator=None,
        grid_h= None,
        grid_w= None,
        ):
        

        """
        Processes a list of images based on the provided text prompts and settings, with optional upsampling to enhance image resolution or detail.

        Parameters:
            images (list[list[Image.Image]]): A collection of images arranged in a grid layout, where each row represents an in-context example or the current query. 
            The current query should be placed in the last row. The target image may be None in the input, while all other images should be of the PIL Image type (Image.Image).
            
            prompts (list[str]): A list containing three prompts: the layout prompt, task prompt, and content prompt, respectively.
            
            seed (int): A fixed integer seed to ensure reproducibility of random elements during processing.
            
            cfg (int): The strength of Classifier-Free Diffusion Guidance, which controls the degree of influence over the generated results.
            
            steps (int): The number of sampling steps to be performed during processing.
            
            upsampling_steps (int): The number of denoising steps to apply when performing upsampling.
            
            upsampling_noise (float): The noise level used as a starting point when upsampling with SDEdit. A higher value reduces noise, and setting it to 1 disables SDEdit, causing the PIL resize function to be used instead.
            
            is_upsampling (bool, optional): A flag indicating whether upsampling should be applied using SDEdit.

        Returns:
            Processed images resulting from the algorithm, with optional upsampling applied based on the `is_upsampling` flag.
        """
        
        # if seed == 0:
        #     seed = random.randint(0, 2 ** 32 - 1)
        
        self.sample_fn = self.sampler.sample_ode(
            sampling_method=self.solver,
            num_steps=steps,
            atol=self.atol,
            rtol=self.rtol,
            reverse=False,
            do_shift=True,
            time_shifting_factor=self.time_shifting_factor,
        )

        # Use class grid size
        # grid_h, grid_w = self.grid_h, self.grid_w
        
        device_type = self.device if isinstance(self.device, str) else self.device.type
        with torch.autocast(enabled=True, device_type=device_type, dtype=torch.bfloat16):   
            with torch.no_grad():
                #inp = prepare_modified(t5=self.t5, clip=self.clip, img=x, prompt=[' '.join(prompts)], proportion_empty_prompts=0.0)
                model_kwargs = dict(
                    txt=inp["txt"], 
                    txt_ids=inp["txt_ids"], 
                    txt_mask=inp["txt_mask"],
                    y=inp["vec"], 
                    img_ids=inp["img_ids"], 
                    img_mask=inp["img_mask"], 
                    cond=img_cond,
                    guidance=torch.full((1,), cfg, device=self.device, dtype=self.dtype),
                )

                print(f"start sampling")
                samples = self.sample_fn(inp["img"], self.model.forward, model_kwargs)[-1]

            # Get query row
            samples = samples[:1]
            row_samples = []
            start = 0
            with torch.no_grad():
                for size in sliced_subimage:
                    end = start + (size[0] * size[1] // 256)
                    latent_h = size[0] // 8
                    latent_w = size[1] // 8
                    row_sample = samples[:, start:end, :]
                    row_sample = rearrange(row_sample.float(), "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=2, pw=2, h=latent_h//2, w=latent_w//2)
                    row_sample = ae.decode(row_sample / ae.scale_factor + ae.shift_factor) #ae.decode(row_sample / ae.config.scaling_factor + ae.config.shift_factor)[0]
                    row_sample = (row_sample + 1.0) / 2.0
                    row_sample.clamp_(0.0, 1.0)
                    row_samples.append(row_sample[0])
                    start = end

            # Convert all samples to PIL images
            output_images = []
            for i,row_sample in enumerate(row_samples):
                output_image = to_pil_image(row_sample.float())
                output_images.append(output_image)
            
            torch.cuda.empty_cache()
            
            ret = []
            ret_w = output_images[-1].width
            ret_h = output_images[-1].height
            
            row_start = (grid_h - 1) * grid_w #(3-1)*2
            row_end = grid_h * grid_w #3*2
            print(f"start upsampling") # mask_position [0, 0, 1]
            for i in range(row_start, row_end):
                # when the image is masked, then output it
                if mask_position[i - row_start] and is_upsampling:
                    cropped = output_images[-1].crop(((i - row_start) * ret_w // grid_w, 0, ((i - row_start) + 1) * ret_w // grid_w, ret_h))
                    #cropped.save(f"cropped{i}.png")
                    # content clip
                    for c in content_instruction:
                        if content_prompt.startswith(c):
                            content_prompt = content_prompt.replace(c, '')
                    content_inp,content_img_cond,latent_h, latent_w=pre_x_noise_clip(clip,ae,generator,cropped,content_prompt,upsampling_noise,self.device,dtype=torch.bfloat16,target_size=upsampling_size)
                    upsampled = self.upsampling(
                        ae,
                        cropped, 
                        upsampling_size, 
                        cfg, 
                        upsampling_steps=upsampling_steps, 
                        upsampling_noise=upsampling_noise, 
                        inp=content_inp,
                        img_cond=content_img_cond,
                        latent_h=latent_h,
                        latent_w=latent_w)
                    ret.append(upsampled)
                elif mask_position[i - row_start]:
                    cropped = output_images[-1].crop(((i - row_start) * ret_w // grid_w, 0, ((i - row_start) + 1) * ret_w // grid_w, ret_h))
                    #cropped.save(f"mask_cropped{i}.png")
                    ret.append(cropped)
            
            return ret
