import gradio as gr
import torch
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    AutoTokenizer
)


# Load the pre-trained model and tokenizer
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the caption generation function
def generate_caption(image):
    # Preprocess the image
    input_image = Image.fromarray(image)
    inputs = feature_extractor(images=input_image, return_tensors="pt")
    
    # Generate caption
    input_ids = inputs["pixel_values"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    generated_ids = model.generate(input_ids, num_beams=4, max_length=16)
    caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return caption


# Create a Gradio interface
title = "Image Caption Generator"
description = "Upload an image and let the model generate a caption for it."
inputs = gr.inputs.Image(label="Input Image")
outputs = gr.outputs.Textbox(label="Caption")
examples = [
    ["images/IMG_0514.JPG"],
    ["images/IMG_0512.JPG"],
    ["images/IMG_2516.JPG"]
]
interface = gr.Interface(fn=generate_caption, inputs=inputs, outputs=outputs, title=title, description=description, examples=examples, capture_session=True)

# Launch the interface
interface.launch()