from diffusers import DiffusionPipeline,AutoPipelineForText2Image, LCMScheduler, AutoencoderTiny
from compel import Compel, ReturnedEmbeddingsType
import torch
from typing import Tuple
# import os

import time
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
# model_id = "stabilityai/stable-diffusion-xl-base-1.0"
model_id = "D:\AIGC\/nightXL"
adapter_id = "latent-consistency/lcm-lora-sdxl"

# pipe = DiffusionPipeline.from_pretrained(model_id,local_files_only=True,torch_dtype=torch.float16,use_safetensors=True)
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



style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt},,ray tracing",
        "negative_prompt": "worst quality, normal quality, low quality, low res",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured,",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly,",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast,",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style,",
    },
    {
        "name": "Oil painting",
        "prompt": "Oil painting, Cartoon of a Sensual athletic Slovak, {prompt}, [art by Paul Lehr, (Bruno Catalano:0.7) ::2], Sepia filter",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic,",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed,",
        "negative_prompt": "photo, photorealistic, realism, ugly,",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic,",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white,",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured,",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting,",
    },
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "(No style)"


def apply_style(style_name: str, positive: str, negative_prompt: str = "") -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + negative_prompt






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
default_prompt = "a very beautiful cyborg made of transparent glossy glass skin surrounded with glowing tubes inside an incubator of a futuristic hospital bio lab,rendered by beeple, by syd meade, by android jones, by yoanne lossel, by artgerm and greg rutkowski,space art concept, sci - fi, digital art, unreal engine, wlop, trending artstation,RAW photo, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
default_negative = "worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting"
default_guidance = 1
default_step = 5
def predict(
    prompt, negative_prompt=default_negative, style_name=None,guidance=default_guidance, steps=default_step, progress=gr.Progress(track_tqdm=True)
):
    # generator = torch.manual_seed(seed)
    if len(prompt) == 0:
        prompt = default_prompt
    prompt,n_prompt = apply_style(style_name, prompt, negative_prompt)
    prompt_embeds, pooled_prompt_embeds = compel_proc(prompt)
    start_time = time.time()
    results = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt=n_prompt,
        # generator=generator,
        num_inference_steps=steps,
        num_images_per_prompt=2,
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
    print(time.time() - start_time)
    return results.images


examples = [
    [
        "a cute Shiba Inu that are standing in the snow,in the winter, warm sun, Christmas decoration , red scarf",
        None,
        None
    ],
    [
        'by Joseph Ducreux, Jerry Siegel,  Pastels artwork,close up of a Eclectic Weary [Ivorian:Chinoiserie:18] (Erik Per Sullivan:1.3) casting a Frost magic spell, fantasy art, magical, mythical, fluorescent magic aura, Magic Mage, dressed in 1970s disco fashion with Sandstone adornments, Action scene, Goblincore deep purple Glasses Chain, inside a Jagged Le Morne Brabant, Hurricane, split diopter, Screen print, Guilty, Qajar Art, Nostalgic lighting, dynamic, Sony A9 II, Depth of field 270mm, Low Contrast, skin pores',
        None,
        None
    ],
    [
        '4 Golden Retriever playfully chasing leaves in an autumn park, with colorful foliage and a sense of calm., Miki Asai Macro photography, close-up, hyper detailed, trending on artstation, sharp focus, studio photo, intricate details, highly detailed, by greg rutkowski',
        None,
        None
    ],
    [
        'Brown haired 28 years old woman is watching to viewer, wearing rose gold glasses (square edge in the top on the outer side), pink thin longsweat, small cleavage, white hexagonical crystal jewelry',
        None,
        None
    ],
    [
        'A Yukisakura masterpiece featuring a quaint dwelling nestled by a tranquil lake, surrounded by an enchanting forest under a pristine blue sky scattered with languid clouds, with lofty mountains in the backdrop, high quality, impressionist style, abstract intonations, the play of sweeping brush strokes, palette knife effects, canvas medium, landscape orientation, interplay of perspective, impasto applications, dramatic chiaroscuro, stark contrasts, harmonious color palette, inviting textures, breathtaking full color',
        None,
        None
    ],
    
]


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
                # negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Negative Prompt", show_label=True, lines=3)
            with gr.Column():
                generate_bt = gr.Button("Generate", variant='primary', scale=1)
        gallery = gr.Gallery(
                label="Generated images", show_label=False, elem_id="gallery", preview=True, columns=[2], rows=[2], object_fit="contain", height="auto"
        )
    
        # image = gr.Image(type="filepath")
        # with gr.Accordion("Advanced options", open=False):
        #     guidance = gr.Slider(
        #         label="Guidance", minimum=0.0, maximum=3, value=1, step=0.001
        #     )
        #     steps = gr.Slider(label="Steps", value=4, minimum=2, maximum=8, step=1)
            # seed = gr.Slider(
            #     randomize=True, minimum=-1, value=-1,maximum=12013012031030, label="Seed", step=1
            # )
        with gr.Accordion("Advanced settings", open=False):
             style_selection = gr.Radio(
                               show_label=True, container=True, interactive=True,
                               choices=STYLE_NAMES,
                               value=DEFAULT_STYLE_NAME,
                               label='Image Style'
             )
             negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Negative Prompt", show_label=True, lines=3)
            #  guidance_scale = gr.Slider(
            #     label="Guidance Scale", minimum=0, maximum=50, value=7.5, step=0.5
            #  )    
        ex = gr.Examples(examples=examples, fn=predict, inputs=[prompt, negative_prompt,style_selection], outputs=gallery, cache_examples=True)
        # negative_prompt.submit(predict, inputs=[prompt, negative_prompt,style_selection], outputs=gallery, postprocess=False)
        # prompt.submit(predict, inputs=[prompt, negative_prompt,style_selection], outputs=gallery, postprocess=False)
        # generate_bt.click(predict, inputs=[prompt, negative_prompt,style_selection], outputs=gallery, postprocess=False)
        inputs = [prompt, negative_prompt,style_selection]
        generate_bt.click(fn=predict, inputs=inputs, outputs=gallery, postprocess=True)

demo.queue()
demo.launch(server_port=6006)
