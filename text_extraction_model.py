# -*- coding: utf-8 -*-
"""text_extraction_model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1sstJhtXqi2x2fHiBP978OjokVpaISSIa
"""

from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import os

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def identify_object(image_path, descriptions):
    image = Image.open(image_path)
    image_inputs = processor(images=image, return_tensors="pt")
    text_inputs = processor(text=descriptions, return_tensors="pt", padding=True)

    # Get image and text features
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        text_features = model.get_text_features(**text_inputs)

    # Compute similarity scores
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    similarity = torch.matmul(image_features, text_features.T)
    most_similar_idx = similarity.argmax().item()

    return descriptions[most_similar_idx]

def identify_objects_in_directory(images_dir, descriptions):
    descriptions_dict = {}
    for filename in os.listdir(images_dir):
        if filename.endswith(".png"):  # Process only PNG files
            image_path = os.path.join(images_dir, filename)
            description = identify_object(image_path, descriptions)
            descriptions_dict[filename] = description
            print(f"Image {filename} identified as: {description}")

    return descriptions_dict

images_dir = 'segmented_objects'  # Directory containing segmented object images
descriptions = [
    "a photo of a person",
    "a photo of a bus",
    "a photo of a car",
    "a photo of a bag"
]

# Identify objects in the images
identifications = identify_objects_in_directory(images_dir, descriptions)
print(identifications)