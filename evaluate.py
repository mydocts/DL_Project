# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import T5Tokenizer, T5EncoderModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from utils.pipeline import Pipeline
from data.calvindataset import CalvinDataset_Goalgen
class IP2PEvaluation(object):
    def __init__(self, 
                 ckpt_path,
                 res=256):    
        # Init models
        pretrained_model_dir = os.path.abspath("resources/IP2P")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.text_encoder = T5EncoderModel.from_pretrained("t5-base")
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_dir, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_dir, subfolder="unet")
        
        # Modify UNet for 20 frames (4 noisy + 20*4 conditioning = 84 channels)
        self.in_channels = 84
        with torch.no_grad():
            # Get original weights
            old_conv_in = self.unet.conv_in
            new_conv_in = torch.nn.Conv2d(
                self.in_channels, old_conv_in.out_channels, 
                kernel_size=old_conv_in.kernel_size, 
                padding=old_conv_in.padding
            )
            
            # Initialize new weights: copy first 8 channels, zero the rest
            new_conv_in.weight[:, :8, :, :] = old_conv_in.weight
            new_conv_in.weight[:, 8:, :, :] = 0
            new_conv_in.bias = old_conv_in.bias
            
            self.unet.conv_in = new_conv_in
            self.unet.config.in_channels = self.in_channels

        # Load weight for unet
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint from {ckpt_path}")
            payload = torch.load(ckpt_path, map_location=self.device)
            state_dict = payload['state_dict']
            del payload
            # Handle strict=False because we might be loading 8-channel weights into 84-channel model if using old ckpt
            # Or if loading new ckpt, it should match.
            # For safety in this transition, we use strict=False and check keys manually if needed
            msg = self.unet.load_state_dict(state_dict['unet_ema'], strict=False)
            print(msg)
        else:
            print(f"Checkpoint {ckpt_path} not found. Using pretrained weights (with expanded channels).")

        # Check if fp16 weights exist
        variant = None
        if os.path.exists(os.path.join(pretrained_model_dir, "unet", "diffusion_pytorch_model.fp16.bin")):
            print("Found fp16 weights. Using variant='fp16'.")
            variant = "fp16"

        self.pipe = Pipeline.from_pretrained(
            pretrained_model_dir,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            vae=self.vae,
            unet=self.unet,
            revision=None,
            variant=variant,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16
        ).to(self.device)

        self.pipe.safety_checker = None
        self.pipe.requires_safety_checker = False
        self.generator = torch.Generator(self.device).manual_seed(42)

        # Diffusion hyparams
        self.num_inference_steps = 50
        self.image_guidance_scale = 2.5
        self.guidance_scale = 7.5

        # Image transform
        self.res = res
        self.transform = transforms.Resize((res, res))

    def evaluate(self, eval_result_dir, eval_data_dir,is_training):
        os.makedirs(eval_result_dir,exist_ok=True)
        
        dataset = CalvinDataset_Goalgen(
            eval_data_dir, 
            resolution=256,
            resolution_before_crop=288,
            center_crop=True,
            forward_n_min_max=(20, 22), 
            is_training=is_training,
            use_full=True,
            color_aug=False
        )
        
        save_count = 0
        for i in range(0, len(dataset), 100):
            example = dataset[i]
            text=example['input_text']
            original_pixel_values = example['original_pixel_values']
            edited_pixel_values = example['edited_pixel_values']
            progress=example["progress"]

            progress=progress*10
            text[0]=text[0]+f".And {progress}% of the instruction has been finished." 
            print(text[0])
            input_image_batch=[original_pixel_values]
            predict_image = self.inference(input_image_batch, text)

            fig, ax = plt.subplots(1,4, figsize=(20, 5))
            fig.suptitle(text[0], fontsize=12)
            
            # Input is (20, 3, H, W)
            # Show first frame (t-19)
            first_frame = original_pixel_values[0].permute(1, 2, 0).numpy()
            first_frame = (first_frame + 1) / 2 * 255
            first_frame = np.clip(first_frame, 0, 255).astype(np.uint8)
            ax[0].imshow(first_frame)
            ax[0].set_title("Input Frame t-19")
            ax[0].axis('off')

            # Show last frame (t)
            last_frame = original_pixel_values[-1].permute(1, 2, 0).numpy()
            last_frame = (last_frame + 1) / 2 * 255
            last_frame = np.clip(last_frame, 0, 255).astype(np.uint8)
            ax[1].imshow(last_frame)
            ax[1].set_title("Input Frame t")
            ax[1].axis('off')

            edited_image = edited_pixel_values.permute(1, 2, 0).numpy()
            edited_image = (edited_image + 1) / 2 * 255
            edited_image = np.clip(edited_image, 0, 255)
            edited_image = edited_image.astype(np.uint8)
            ax[2].imshow(edited_image)
            ax[2].set_title("Ground Truth")
            ax[2].axis('off')

            ax[3].imshow(predict_image[0])
            ax[3].set_title("Predicted")
            ax[3].axis('off')

            save_path = os.path.join(eval_result_dir, f"debug_{save_count}.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            save_count += 1
            print("Test run complete. Saved one sample.")
            break

    def inference(self, image_batch, text_batch):
        """Inference function."""
        input_images = []
        for image in image_batch:
            if isinstance(image, np.ndarray):
                image=Image.fromarray(image)
            input_image = self.transform(image)
            input_images.append(input_image)
        edited_images = self.pipe(
            prompt=text_batch,
            image=input_images,
            num_inference_steps=self.num_inference_steps,
            image_guidance_scale=self.image_guidance_scale,
            guidance_scale=self.guidance_scale,
            generator=self.generator,
            safety_checker=None,
            requires_safety_checker=False).images
        edited_images=[ np.array(image) for image in edited_images]

        return edited_images

if __name__ == "__main__":  
    ckpt_path="dummy_ckpt.ckpt"
    eval = IP2PEvaluation(ckpt_path)
    eval_data_dir = os.path.abspath("dummy_data")
    eval_result_dir = "debug_vis"
    eval.evaluate(eval_result_dir, eval_data_dir,is_training=False)