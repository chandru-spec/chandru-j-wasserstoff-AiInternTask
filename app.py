import streamlit as st
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image

# Load the pre-trained model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load and preprocess the image
def preprocess_image(image):
    image_rgb = np.array(image.convert('RGB'))
    image_tensor = F.to_tensor(image_rgb).unsqueeze(0)
    return image_tensor, image

# Perform segmentation
def segment_objects(image_tensor, image):
    with torch.no_grad():
        prediction = model(image_tensor)

    masks = prediction[0]['masks'].cpu().numpy()
    boxes = prediction[0]['boxes'].cpu().numpy()

    image = np.array(image)
    for i in range(len(masks)):
        mask = masks[i, 0]
        mask = (mask > 0.5).astype(np.uint8)
        color = np.random.randint(0, 255, size=(3,))
        image[mask == 1] = image[mask == 1] * 0.5 + color * 0.5

    return image

def main():
    st.title('Object Detection with Mask R-CNN')
    
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_tensor, image = preprocess_image(image)
        segmented_image = segment_objects(image_tensor, image)
        
        st.image(segmented_image, caption='Segmented Image', use_column_width=True)

if __name__ == "__main__":
    main()
