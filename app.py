import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import os
from rembg import remove
from models.style_transfer import StyleTransfer

# Initialize session state variables if they don't exist
if 'content_image_pil' not in st.session_state:
    st.session_state.content_image_pil = None
if 'content_image_path' not in st.session_state:
    st.session_state.content_image_path = None
if 'bg_removed' not in st.session_state:
    st.session_state.bg_removed = False
if 'original_content_image' not in st.session_state:
    st.session_state.original_content_image = None

# Function to remove background from images
def remove_background(image):
    """Remove background from an image using rembg library.
    
    Args:
        image: PIL Image object
        
    Returns:
        PIL Image with background removed and alpha channel
    """
    # Convert PIL image to numpy array
    np_image = np.array(image)
    
    # Remove background - this creates an RGBA image with transparency
    output = remove(np_image)
    
    # Return the RGBA image directly without forcing RGB conversion
    output_pil = Image.fromarray(output)
    
    # Handle image with alpha channel conversion to RGB if needed
    if output_pil.mode == 'RGBA':
        # Create a white background
        background = Image.new('RGB', output_pil.size, (255, 255, 255))
        # Composite the image with alpha over the background
        background.paste(output_pil, mask=output_pil.split()[3])
        return background
    else:
        return output_pil

# Set page configuration
st.set_page_config(
    page_title="Neural Style Transfer",
    page_icon="🎨",
    layout="wide"
)

# Create directories if they don't exist
os.makedirs("documents", exist_ok=True)

# Initialize the StyleTransfer model
@st.cache_resource
def load_model():
    return StyleTransfer()

# App title and description
st.title("Neural Style Transfer App")
st.markdown("""
This app applies neural style transfer using TensorFlow Lite models.
Upload a content image (background will be automatically removed) and a style image to apply.
""")

# Content image upload section
st.subheader("Step 1: Upload Content Image")
content_upload = st.file_uploader("Upload content image (what will be stylized)", type=["jpg", "jpeg", "png"],
                                 key="content_upload")

if content_upload is not None:
    # Save original image for comparison
    original_image = Image.open(content_upload).convert("RGB")
    st.session_state.original_content_image = original_image
    
    # Process the uploaded image and automatically remove background
    with st.spinner("Removing background... This may take a moment."):
        # Apply background removal
        st.session_state.content_image_pil = remove_background(original_image)
        
        # Save the no background image
        no_bg_path = "documents/no_bg_photo.png"
        st.session_state.content_image_pil.save(no_bg_path)
        st.session_state.content_image_path = no_bg_path
        st.session_state.bg_removed = True
    
    # Display original and background-removed images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original Content Image", use_container_width=True)
    with col2:
        st.image(st.session_state.content_image_pil, caption="Background Removed", use_container_width=True)
else:
    st.info("Please upload a content image. Background will be automatically removed.")
    st.session_state.content_image_pil = None
    st.session_state.content_image_path = None
    st.session_state.bg_removed = False

# Style image upload section
st.subheader("Step 2: Upload Style Image")
style_upload = st.file_uploader("Upload your style image (how you want it to look)", type=["jpg", "jpeg", "png"])

selected_style_path = None
if style_upload is not None:
    # User uploaded a style image
    style_image_pil = Image.open(style_upload).convert("RGB")
    style_image_path = "documents/style_image.jpg"
    style_image_pil.save(style_image_path)
    st.image(style_image_pil, caption="Uploaded Style Image", use_column_width=True, clamp=False)
    selected_style_path = style_image_path
else:
    # Optional default style selection
    st.info("Please upload a style image that will determine how your content will look after processing.")

# Style transfer parameters
if st.session_state.content_image_pil is not None and selected_style_path is not None:
    st.subheader("Step 3: Adjust Style Transfer Parameters")
    content_blending_ratio = st.slider(
        "Content-Style Blending Ratio (higher preserves more content)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.1
    )

    # Processing section
    st.subheader("Step 4: Process Images")

    if st.button("Apply Style Transfer"):
        with st.spinner("Initializing style transfer model... This may take a moment."):
            # Initialize the model
            model = load_model()
        
        with st.spinner("Processing images... This may take a minute."):
            # Apply style transfer
            result_pil = model.transfer_style(
                st.session_state.content_image_pil,
                selected_style_path,
                content_blending_ratio
            )
            
            # Save the stylized image
            result_path = "documents/stylized_image.jpg"
            result_pil.save(result_path)
            
            # Show results
            st.subheader("Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(st.session_state.content_image_pil, caption="Content Image (No Background)", use_container_width=True)
            
            with col2:
                st.image(style_image_pil, caption="Style Image", use_column_width=True, clamp=False)
            
            with col3:
                st.image(result_pil, caption="Stylized Result", use_container_width=True)
                
            # Download button
            buf = io.BytesIO()
            result_pil.save(buf, format="JPEG")
            buf.seek(0)
            
            st.download_button(
                label="Download Stylized Image",
                data=buf,
                file_name="stylized_image.jpg",
                mime="image/jpeg"
            )

# Footer with additional information
st.markdown("---")
st.markdown("""
### About this app
This application uses TensorFlow Lite to apply neural style transfer.

**Process:**
1. Upload your content image (background is automatically removed)
2. Upload your style image (this determines the artistic style to apply)
3. Adjust the blending ratio to control style transfer intensity
4. Run the style transfer process

**Technologies used:**
- TensorFlow Lite for model inference
- Streamlit for the web interface
- Magenta's Arbitrary Image Stylization model
- Rembg for automatic background removal
""")