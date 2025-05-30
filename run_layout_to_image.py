import os
import cv2
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from utils.generation_utils import load_checkpoint, bbox_encode, draw_layout

########################
# Set random seed
#########################
from accelerate.utils import set_seed
set_seed(0)

def run_layout_to_image(layout, args):
  ########################
  # Build pipeline
  #########################
  pipe, generation_config = load_checkpoint(args.ckpt_path)
  pipe = pipe.to("cuda")
  args = {arg: getattr(args, arg) for arg in vars(args) if getattr(args, arg) is not None}
  generation_config.update(args)
  
  # Sometimes the nsfw checker is confused by the Pokémon images, you can disable
  # it at your own risk here
  disable_safety = True
  if disable_safety:
    def null_safety(images, **kwargs):
        return images, False
    pipe.safety_checker = null_safety
  
  ########################
  # Encode layout and build text prompt
  #########################
  # timeofday and weather sanity check  
  assert not generation_config['dataset'] == 'nuimages' or "timeofday" not in layout or layout['timeofday'] in ['daytime', 'night']
  assert not generation_config['dataset'] == 'nuimages' or "weather" not in layout or layout['weather'] in ['sunny', 'rain']
  if "timeofday" in generation_config['prompt_template'] and "timeofday" not in layout.keys():
    layout["timeofday"] = "daytime"
  if "weather" in generation_config['prompt_template'] and "weather" not in layout.keys():
    layout["weather"] = "sunny"

  # camera sanity check
  assert not generation_config['dataset'] == 'nuimages' or ("camera" in layout and layout['camera'] in ['front', 'front left', 'front right', 'back', 'back left', 'back right'])
  bboxes = layout['bbox'].copy()
  layout["bbox"] = bbox_encode(layout['bbox'], generation_config)
  prompt = generation_config['prompt_template'].format(**layout)
  print(prompt)
  
  ########################
  # Generation
  ########################
  # generation params
  width = generation_config["width"]  
  height = generation_config["height"]
  scale = generation_config["cfg_scale"]
  n_samples = generation_config["nsamples"]
  num_inference_steps = generation_config["num_inference_steps"]
  
  # run generation
  images = pipe(n_samples*[prompt], guidance_scale=scale, num_inference_steps=num_inference_steps, height=int(height), width=int(width)).images
  
  ########################
  # Save results
  #########################
  root = args["output_dir"]
  os.makedirs(root, exist_ok=True)
  layout_canvas = draw_layout(bboxes)
  layout_canvas = Image.fromarray(layout_canvas, mode='RGB').save(os.path.join(root, '{}_layout.jpg'.format(generation_config['dataset'])))
  for idx, image in enumerate(images):
    image = np.asarray(image)
    image = Image.fromarray(image, mode='RGB')
    image.save(os.path.join(root, '{}_{}.jpg'.format(generation_config['dataset'], idx)))

if __name__ == "__main__":
  parser = ArgumentParser(description='Layout-to-image generation script')
  parser.add_argument('ckpt_path', type=str)
  parser.add_argument('--nsamples', type=int, default=1)
  parser.add_argument('--cfg_scale', type=float, default=None)
  parser.add_argument('--num_inference_steps', type=int, default=None)
  parser.add_argument('--output_dir', type=str, default="./results/")
  args = parser.parse_args()
  
  ########################
  # Define layouts
  # Note: 
  # 1) "camera": specific for nuimages, and should be selected from [front, front left, front right, back, back left, back right]
  # 2) "bbox": list of bounding boxes, each defined as [category, x1, y1, x2, y2] 
  #   a) category (str):, check dataset2classes in utils.generation_utils
  #   b) x1, y1, x2, y2 (float): in range of [0, 1]
  ########################
  # example layout for nuimages
  # layout = {
  #   "camera": "front",
  #   "bbox": [
  #     ["car", 0.756875, 0.4622, 0.90375, 0.5844],
  #     ["car", 0.47625, 0.4822, 0.691875, 0.8011],
  #     ["pedestrian", 0.0, 0.4933, 0.223125, 0.9267],
  #     ["car", 0.273125, 0.4511, 0.47375, 0.6444],
  #     ["car", 0.7125, 0.6689, 0.999375, 1.0],
  #   ]
  # }
  
  # # example layout for nuimages with timeofday and weather
  # layout = {
  #   "camera": "front",
  #   "timeofday": "night",
  #   "weather": "rain",
  #   "bbox": [
  #     ["car", 0.756875, 0.4622, 0.90375, 0.5844],
  #     ["car", 0.47625, 0.4822, 0.691875, 0.8011],
  #     ["car", 0.0, 0.4933, 0.223125, 0.9267],
  #     ["car", 0.273125, 0.4511, 0.47375, 0.6444],
  #     ["car", 0.7125, 0.6689, 0.999375, 1.0],
  #   ]
  # }
  
  # # example layout for coco-stuff
  layout = {
    "bbox": [
      ["pizza", 0.6809, 0.6719, 0.9461, 0.9056],
      ["pizza", 0.6711, 0.3612, 0.9717, 0.6177],
      ["pizza", 0.3618, 0.6787, 0.6629, 0.9775],
      ["pizza", 0.0697, 0.6652, 0.2854, 0.9416],
      ["pizza", 0.0329, 0.0212, 0.6471, 0.5388],
      ["pizza", 0.6726, 0.0315, 0.9714, 0.2954],
    ]

  }
  
  ########################
  # Run layout-to-image generation
  ########################
  run_layout_to_image(layout, args)