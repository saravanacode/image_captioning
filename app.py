import torch
from transformers import AutoProcessor,AutoModelForCausalLM
import gradio as gr 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor=AutoProcessor.from_pretrained("alibidaran/General_image_captioning")
model=AutoModelForCausalLM.from_pretrained("alibidaran/General_image_captioning").to(device)
def generate_caption(image):
  encoded=processor(images=image, return_tensors="pt").to(device)
  pixels=encoded['pixel_values'].to(device)
  with torch.no_grad():
    generated_ids=model.generate(pixel_values=pixels,max_length=10)
  generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
  return generated_caption 
demo=gr.Interface(
    fn=generate_caption,
    inputs=[
        gr.Image(type='pil'),
    ],
    outputs= 'label',
    examples=['sample.jpg','sample1.jpg','sample2.jpg'],
      theme=gr.themes.Soft(primary_hue='purple',secondary_hue=gr.themes.colors.gray)
)
demo.launch(show_error=True)