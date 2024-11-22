"""
Automated Image Tagging System: A Streamlit application for automated image captioning and object detection 
using a pre-trained AI model from Hugging Face.
"""

import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import matplotlib.pyplot as plt

# Set the page layout for the Streamlit app
st.set_page_config(
    page_title="Automated Image Tagging System",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="auto",
)

# Define the header and description for the Streamlit app
st.title("Automated Image Tagging System")
st.markdown("""
Welcome to the **Automated Image Tagging System**. This project allows you to upload images and 
automatically generate captions and detect objects within the image using a powerful AI model.

Simply upload an image below to get started!
""")

# Function to load the model and processor
@st.cache_resource
def load_model():
    """
    Load the pre-trained Florence-2-large model and processor from Hugging Face.

    Returns:
        model: Loaded Hugging Face model.
        processor: Loaded Hugging Face processor for pre-processing.
    """
    model_id = 'microsoft/Florence-2-large'  # Replace this if the model ID changes
    loaded_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval()
    loaded_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return loaded_model, loaded_processor

# Load the model and processor only once (cached for efficiency)
model_instance, processor_instance = load_model()

# Function to process image and generate captions
def process_image(uploaded_image):
    """
    Process the uploaded image and generate a caption.

    Args:
        uploaded_image (PIL.Image.Image): The uploaded image to process.

    Returns:
        generated_text (str): The generated caption for the image.
    """
    task_prompt = '<CAPTION>'  # Example prompt for image captioning
    inputs = processor_instance(text=task_prompt, images=uploaded_image, return_tensors="pt")
    
    with torch.no_grad():
        generated_ids = model_instance.generate(
            input_ids=inputs["input_ids"],  # Provide the input text IDs
            pixel_values=inputs["pixel_values"],  # Provide the pixel values from the image
            max_new_tokens=1024,
            early_stopping=True,
            num_beams=3,
        )
    
    generated_text = processor_instance.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# Function for object detection
def object_detection(uploaded_image):
    """
    Perform object detection on the uploaded image.

    Args:
        uploaded_image (PIL.Image.Image): The uploaded image to process.

    Returns:
        detection_results (str): Detected objects in the form of a string (labels and possibly bounding boxes).
    """
    task_prompt = '<OD>'  # Example prompt for object detection
    inputs = processor_instance(text=task_prompt, images=uploaded_image, return_tensors="pt")
    
    with torch.no_grad():
        generated_ids = model_instance.generate(
            input_ids=inputs["input_ids"],  # Provide the input text IDs
            pixel_values=inputs["pixel_values"],  # Provide the pixel values from the image
            max_new_tokens=1024,
            early_stopping=True,
            num_beams=3,
        )
    
    detection_results = processor_instance.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return detection_results

# Function to parse object detection results
def parse_detection_results(detection_results):
    """
    Parse the object detection results string into a list of detected objects.

    Args:
        detection_results (str): The raw object detection string results.

    Returns:
        parsed_labels (list): A list of detected object labels.
    """
    parsed_labels = detection_results.split(",")  # Adjust parsing logic as needed
    return parsed_labels

# Function to plot bounding boxes for object detection (dummy for now)
def plot_bbox(uploaded_image, detected_labels):
    """
    Display the image and detected object labels. (Bounding box logic can be added later)

    Args:
        uploaded_image (PIL.Image.Image): The uploaded image.
        detected_labels (list): List of detected object labels.
    """
    fig, ax = plt.subplots()
    ax.imshow(uploaded_image)
    
    # For now, we just display the labels at random positions (as bounding boxes are not available)
    for i, label in enumerate(detected_labels):
        plt.text(10, 30 * (i + 1), label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
    
    ax.axis('off')  # Hide the axes
    st.pyplot(fig)

# Uploading image section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Check if an image is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)

    # Process the uploaded image for captioning
    st.write("Processing image... ⏳")
    generated_caption = process_image(uploaded_image)

    # Display the output caption
    st.write("**Generated Tags**:")
    st.success(generated_caption)

    # Perform object detection and display results
    st.write("Performing object detection... ⏳")
    detection_results = object_detection(uploaded_image)
    st.write("**Raw Object Detection Results**:")
    st.write(detection_results)

    # Parse and display object detection results
    parsed_labels = parse_detection_results(detection_results)
    st.write("**Parsed Object Detection Labels**:")
    st.json(parsed_labels)

    # Plot labels (replace with bounding boxes if available)
    plot_bbox(uploaded_image, parsed_labels)
else:
    st.warning("Please upload an image to continue.")

# Footer information
st.markdown("""
    ---
    **Project created by DEPI AWS_ML_GradTeam**  
    Powered by [Hugging Face](https://huggingface.co/)
""")
