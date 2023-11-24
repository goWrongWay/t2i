from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import time
from diffusers import LCMScheduler, AutoPipelineForText2Image
import os
app = Flask(__name__)

# Load models and initialize pipeline
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
adapter_id = "latent-consistency/lcm-lora-sdxl"

pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
pipe.load_lora_weights(adapter_id)
pipe.fuse_lora()

# Front-end route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/public/generated_images/<filename>')
def serve_generated_image(filename):
    return send_from_directory('public/generated_images', filename)

@app.route('/generate_image', methods=['POST'])
def generate_image():
    try:
        # Get the input prompt from the request
        data = request.get_json() 
        prompt = data.get('prompt', '')
         # Generate a random filename
         # 获取当前时间戳（以秒为单位）
        current_milliseconds = int(time.time()) * 1000
        
        filename = str(current_milliseconds) + '.png'

        # Save the image with the random filename
        image_path = os.path.join('public/generated_images', filename)
        image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=1.2).images[0]
        image.save(image_path)

        return jsonify({'success': True, 'image': filename,})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(port=6016)