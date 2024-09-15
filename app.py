from transformers import CLIPFeatureExtractor
from safety_checker import StableDiffusionSafetyChecker
import torch
from PIL import Image
import gradio as gr
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
safety_checker = StableDiffusionSafetyChecker.from_pretrained(
    "CompVis/stable-diffusion-safety-checker"
).to(device)
feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")


import gradio as gr


def image_classifier(files):
    images = [Image.open(file).convert("RGB").resize((512, 512)) for file in files]

    safety_checker_input = feature_extractor(images, return_tensors="pt").to(device)
    has_nsfw_concepts = safety_checker(
        images=[images], clip_input=safety_checker_input.pixel_values.to(torch.float16)
    )
    results = [
        {"has_nsfw": nsfw, "file": Path(file).name}
        for (nsfw, file) in zip(has_nsfw_concepts, files)
    ]
    return {"results": results}


demo = gr.Interface(
    title="Stable Diffusion Safety Checker API",
    fn=image_classifier,
    inputs=gr.File(file_count="multiple", file_types=["image"]),
    outputs="json",
    api_name="classify",
)
demo.launch()