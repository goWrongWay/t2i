from diffusers import DiffusionPipeline,AutoPipelineForText2Image, LCMScheduler, AutoencoderTiny
from compel import Compel, ReturnedEmbeddingsType
import torch
# import os

# try:
#     import intel_extension_for_pytorch as ipex
# except:
#     pass

# from PIL import Image
# import numpy as np
import gradio as gr
# import psutil

# SAFETY_CHECKER = os.environ.get("SAFETY_CHECKER", None)
# TORCH_COMPILE = os.environ.get("TORCH_COMPILE", None)
# check if MPS is available OSX only M1/M2/M3 chips
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
device = torch.device(
    "cuda" if torch.cuda.is_available() else "xpu" if xpu_available else "cpu"
)
torch_device = device
torch_dtype = torch.float16

# print(f"SAFETY_CHECKER: {SAFETY_CHECKER}")
# print(f"TORCH_COMPILE: {TORCH_COMPILE}")
print(f"device: {device}")

if mps_available:
    device = torch.device("mps")
    torch_device = "cpu"
    torch_dtype = torch.float32

# model_id = "stabilityai/stable-diffusion-xl-base-1.0"

# if SAFETY_CHECKER == "True":
    # pipe = DiffusionPipeline.from_pretrained(model_id)
# else:
    # pipe = DiffusionPipeline.from_pretrained(model_id, safety_checker=None)
# pipe = DiffusionPipeline.from_pretrained(model_id)
# pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
# pipe.to(device=torch_device, dtype=torch_dtype).to(device)
# pipe.unet.to(memory_format=torch.channels_last)

# Load models and initialize pipeline
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
adapter_id = "latent-consistency/lcm-lora-sdxl"

pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
pipe.load_lora_weights(adapter_id)
pipe.fuse_lora()

# check if computer has less than 64GB of RAM using sys or os
# if psutil.virtual_memory().total < 64 * 1024**3:
#     pipe.enable_attention_slicing()

# if TORCH_COMPILE:
#     pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
#     pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)

#     pipe(prompt="warmup", num_inference_steps=1, guidance_scale=8.0)


# Load LCM LoRA
pipe.load_lora_weights(
    "latent-consistency/lcm-lora-sdxl"
)

compel_proc = Compel(
    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
    requires_pooled=[False, True],
)
default_prompt = "a cute cat, 8k"
default_negative = "worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting"
default_guidance = 1
default_step = 5
def predict(
    prompt=default_prompt, negative_prompt=default_negative, guidance=default_guidance, steps=default_step, progress=gr.Progress(track_tqdm=True)
):
    # generator = torch.manual_seed(seed)
    prompt_embeds, pooled_prompt_embeds = compel_proc(prompt)

    results = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt=negative_prompt,
        # generator=generator,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=1024,
        height=1024,
        # original_inference_steps=params.lcm_steps,
        output_type="pil",
    )
    # nsfw_content_detected = (
    #     results.nsfw_content_detected[0]
    #     if "nsfw_content_detected" in results
    #     else False
    # )
    # if nsfw_content_detected:
    #     raise gr.Error("NSFW content detected.")
    return results.images[0]


css = """
#container{
    margin: 0 auto;
}
#intro{
    max-width: 100%;
    text-align: center;
    margin: 0 auto;
}
"""
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="container"):
        gr.Markdown(
            """# SDXL in 4 steps with Latent Consistency LoRAs
            SDXL is loaded with a LCM-LoRA, giving it the super power of doing inference in as little as 4 steps. 
            """,
            elem_id="intro",
        )
        with gr.Row():
            with gr.Column(scale=6, min_width=600):
                prompt = gr.Textbox(label="Prompt", placeholder="a cute cat, 8k", show_label=True,lines=3)
                negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Negative Prompt", show_label=True, lines=3)
            with gr.Column():
                generate_bt = gr.Button("Generate", variant='primary', scale=1)
        
        image = gr.Image(type="filepath")
        # with gr.Accordion("Advanced options", open=False):
        #     guidance = gr.Slider(
        #         label="Guidance", minimum=0.0, maximum=3, value=1, step=0.001
        #     )
        #     steps = gr.Slider(label="Steps", value=4, minimum=2, maximum=8, step=1)
            # seed = gr.Slider(
            #     randomize=True, minimum=-1, value=-1,maximum=12013012031030, label="Seed", step=1
            # )
            
        inputs = [prompt, negative_prompt]
        generate_bt.click(fn=predict, inputs=inputs, outputs=image)

demo.queue()
demo.launch()
