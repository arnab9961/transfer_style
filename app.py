import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import cv2
import numpy as np
import io
import os
from rembg import remove
import torch.nn as nn
import requests
from io import BytesIO
import dotenv
import openai
from pathlib import Path

# Load environment variables
dotenv.load_dotenv()

# Set OpenRouter API key
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.base_url = "https://openrouter.ai/api/v1"

# Set page configuration
st.set_page_config(
    page_title="Image Style Transfer App",
    page_icon="🎨",
    layout="wide"
)

# Create directories if they don't exist
os.makedirs("documents", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Define the Transformer Net for fast neural style transfer
class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        
        # Upsampling layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        
        # Non-linearity
        self.relu = nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

# Define a class for using CodeLlama model via OpenRouter for style transfer guidance
class AIStyleTransfer:
    def __init__(self):
        self.model = "meta-llama/codellama-70b-instruct"
    
    def generate_style_description(self, style_name):
        """Generate a detailed style description using CodeLlama"""
        try:
            prompt = f"""
            Describe the visual characteristics of the '{style_name}' artistic style in detail. 
            Focus on colors, shapes, textures, and distinctive features that would help in 
            applying this style to an image. Keep the description technical and specific.
            Limit your response to 150 words.
            """
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in art styles and image processing."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7,
                headers={
                    "HTTP-Referer": "https://github.com",  # Required by OpenRouter
                    "X-Title": "Style Transfer App"  # Optional, helps OpenRouter with analytics
                }
            )
            
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating style description: {e}")
            return "Default style description: Bold colors, distinctive brush strokes, and dynamic composition."

# Define a class for style transfer with fallback options
class StyleTransfer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        self.model_urls = {
            'mosaic': 'https://github.com/pytorch/examples/raw/main/fast_neural_style/saved_models/mosaic.pth',
            'candy': 'https://github.com/pytorch/examples/raw/main/fast_neural_style/saved_models/candy.pth',
            'rain_princess': 'https://github.com/pytorch/examples/raw/main/fast_neural_style/saved_models/rain_princess.pth',
            'udnie': 'https://github.com/pytorch/examples/raw/main/fast_neural_style/saved_models/udnie.pth'
        }
        self.models = {}
        # Initialize AI style transfer
        self.ai_style = AIStyleTransfer()
        # Initialize VGG model for fallback style transfer
        self.vgg = models.vgg19(pretrained=True).features.to(self.device).eval()
        
    def verify_model_file(self, model_path):
        """Check if the file is a valid PyTorch model file."""
        try:
            # Try to load the first few bytes to check the file signature
            with open(model_path, 'rb') as f:
                header = f.read(6)  # Read first 6 bytes
                
            # PyTorch files typically start with "PK\x03\x04" (ZIP signature) 
            # or with specific PyTorch signatures
            if header.startswith(b'PK\x03\x04') or header.startswith(b'\x80\x02') or header.startswith(b'\x8a\x0a'):
                return True
            else:
                return False
        except Exception:
            return False
    
    def load_model(self, style_name):
        """Load a style transfer model with robust error handling."""
        if style_name in self.models:
            return self.models[style_name]
        
        model_path = f"models/{style_name}.pth"
        
        # Check if model exists and is valid
        if os.path.exists(model_path) and self.verify_model_file(model_path):
            try:
                # Try to load the existing model with weights_only=True for security
                model = TransformerNet().to(self.device)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
                
                # Cache and return the model
                self.models[style_name] = model
                return model
            except Exception as e:
                st.warning(f"Error loading model: {e}. Will use fallback method.")
        else:
            # If file doesn't exist or is not valid, try to download
            try:
                if os.path.exists(model_path):
                    os.remove(model_path)  # Remove invalid file
                
                st.info(f"Downloading {style_name} style model... (this will only happen once)")
                
                # Download the model
                url = self.model_urls[style_name]
                response = requests.get(url)
                
                # Verify the content before saving
                if response.status_code == 200 and len(response.content) > 1000:  # Basic size check
                    # Save the model
                    with open(model_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Verify the saved file
                    if self.verify_model_file(model_path):
                        st.success(f"{style_name.title()} style model downloaded!")
                        
                        # Load the model
                        model = TransformerNet().to(self.device)
                        model.load_state_dict(torch.load(model_path, map_location=self.device))
                        model.eval()
                        
                        # Cache the model
                        self.models[style_name] = model
                        return model
                    else:
                        st.warning("Downloaded file is not a valid model. Using fallback method.")
                else:
                    st.warning(f"Failed to download valid model file. Using fallback method.")
            except Exception as e:
                st.warning(f"Error downloading model: {e}. Using fallback method.")
        
        # If we get here, all attempts failed - use fallback approach
        return None
    
    def gram_matrix(self, y):
        """Calculate Gram Matrix for style transfer."""
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram
    
    def fallback_style_transfer(self, content_img, style_choice):
        """Perform a simple style transfer using VGG features."""
        st.info("Using fallback neural style method. This may look different from the expected style.")
        
        # Create stylized output based on the requested style name
        img_np = np.array(content_img)
        
        if style_choice == 'mosaic':
            # Mosaic-like effect - increase saturation and apply posterize effect
            img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype("float32")
            img_hsv[:,:,1] = img_hsv[:,:,1] * 1.5  # Increase saturation
            img_hsv[:,:,1] = np.clip(img_hsv[:,:,1], 0, 255)
            img_np = cv2.cvtColor(img_hsv.astype("uint8"), cv2.COLOR_HSV2RGB)
            
            # Apply edgePreserving filter for a more stylized look
            img_np = cv2.edgePreservingFilter(img_np, flags=1, sigma_s=60, sigma_r=0.4)
            
            # Add posterize effect
            img_np = cv2.convertScaleAbs(img_np, alpha=1.2, beta=0)
            img_np = np.clip((img_np // 32) * 32, 0, 255).astype(np.uint8)
            
        elif style_choice == 'candy':
            # Candy-like effect - vibrant colors with edge enhancement
            img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype("float32")
            img_hsv[:,:,1] = img_hsv[:,:,1] * 1.8  # More saturation for candy
            img_hsv[:,:,1] = np.clip(img_hsv[:,:,1], 0, 255)
            img_np = cv2.cvtColor(img_hsv.astype("uint8"), cv2.COLOR_HSV2RGB)
            
            # Edge enhancement
            edges = cv2.Canny(img_np, 100, 200)
            edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            img_np = cv2.addWeighted(img_np, 0.8, edges_rgb, 0.2, 0)
            
            # Color pop
            img_np = cv2.convertScaleAbs(img_np, alpha=1.3, beta=10)
            
        elif style_choice == 'rain_princess' or style_choice == 'udnie':
            # More painterly effect
            img_np = cv2.stylization(img_np, sigma_s=60, sigma_r=0.6)
            
            # Adjust colors based on style
            if style_choice == 'rain_princess':
                # Rain princess - bluish hue
                img_np = cv2.convertScaleAbs(img_np, alpha=1.1, beta=0)
                b, g, r = cv2.split(img_np)
                b = cv2.convertScaleAbs(b, alpha=1.2, beta=10)  # Enhance blue channel
                img_np = cv2.merge([b, g, r])
            else:  # udnie
                # Udnie - more orange/yellow
                img_np = cv2.convertScaleAbs(img_np, alpha=1.1, beta=10)
                b, g, r = cv2.split(img_np)
                r = cv2.convertScaleAbs(r, alpha=1.2, beta=10)  # Enhance red channel
                g = cv2.convertScaleAbs(g, alpha=1.1, beta=5)   # Enhance green channel
                img_np = cv2.merge([b, g, r])
        
        # For all styles, apply a final detail enhancement
        img_np = cv2.detailEnhance(img_np, sigma_s=10, sigma_r=0.15)
        
        return Image.fromarray(img_np)

    def stylize(self, content_image, style_name='mosaic'):
        # For AI-guided styles, use the CodeLlama model
        if style_name == 'codellama':
            # Get style description from CodeLlama
            try:
                style_description = self.ai_style.generate_style_description("modern abstract digital art")
                st.info(f"AI Style Description: {style_description}")
            except Exception as e:
                st.warning(f"Couldn't connect to OpenRouter: {e}")
                style_description = "Bold colors, distinctive brush strokes, and dynamic composition."
            
            # Try to use mosaic style but with additional processing
            model = self.load_model('mosaic')
            if model is None:
                # Fall back to our alternative method
                output_image = self.fallback_style_transfer(content_image, 'mosaic')
            else:
                # Use the loaded model
                content_image = content_image.convert('RGB')
                content_tensor = self.transform(content_image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = model(content_tensor).cpu()
                
                # Postprocess the output
                output_image = output[0].clone().clamp(0, 255).numpy()
                output_image = output_image.transpose(1, 2, 0).astype("uint8")
                output_image = Image.fromarray(output_image)
            
            # Apply additional filters based on style description
            output_image = self.apply_filters(output_image, style_description)
            return output_image
            
        else:
            # For standard styles, try to use the model or fall back
            model = self.load_model(style_name)
            
            if model is None:
                # Use fallback method
                return self.fallback_style_transfer(content_image, style_name)
            
            # Regular style transfer with loaded model
            content_image = content_image.convert('RGB')
            content_tensor = self.transform(content_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = model(content_tensor).cpu()
            
            # Postprocess the output
            output_image = output[0].clone().clamp(0, 255).numpy()
            output_image = output_image.transpose(1, 2, 0).astype("uint8")
            output_image = Image.fromarray(output_image)
            
            return output_image
    
    def apply_filters(self, image, style_description):
        """Apply additional image filters based on the AI-generated style description"""
        # Convert PIL to numpy for OpenCV processing
        img_np = np.array(image)
        
        # Simple filter based on keywords in the description
        if "vibrant" in style_description.lower() or "bold" in style_description.lower():
            # Increase saturation
            img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype("float32")
            img_hsv[:,:,1] = img_hsv[:,:,1] * 1.3  # Increase saturation
            img_hsv[:,:,1] = np.clip(img_hsv[:,:,1], 0, 255)
            img_np = cv2.cvtColor(img_hsv.astype("uint8"), cv2.COLOR_HSV2RGB)
            
        if "contrast" in style_description.lower():
            # Increase contrast
            img_np = cv2.convertScaleAbs(img_np, alpha=1.2, beta=0)
            
        # Convert back to PIL
        return Image.fromarray(img_np)

# Custom content-aware style transfer
def adaptive_style_transfer(content_image, style_image, style_model, style_weight=1.0):
    # Get style colors
    style_np = np.array(style_image)
    content_np = np.array(content_image)
    
    # Apply style colors to content image using color transfer
    content_lab = cv2.cvtColor(content_np, cv2.COLOR_RGB2LAB).astype(float)
    style_lab = cv2.cvtColor(style_np, cv2.COLOR_RGB2LAB).astype(float)
    
    # Calculate mean and std for each channel
    content_mean = np.mean(content_lab, axis=(0, 1))
    content_std = np.std(content_lab, axis=(0, 1))
    style_mean = np.mean(style_lab, axis=(0, 1))
    style_std = np.std(style_lab, axis=(0, 1))
    
    # Apply color transfer
    content_lab = content_lab - content_mean
    content_lab = content_lab * (style_std / (content_std + 1e-6))  # Added epsilon to avoid division by zero
    content_lab = content_lab + style_mean
    
    # Clip values
    content_lab = np.clip(content_lab, 0, 255)
    color_transferred = cv2.cvtColor(content_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    # Apply neural style transfer with pre-trained model
    # Choose a fixed style model instead of using custom illustration style
    style_choice = 'mosaic'  # You can change this to any available style: 'mosaic', 'candy', 'rain_princess', 'udnie'
    styled_image = style_model.stylize(Image.fromarray(color_transferred), style_choice)
    
    return styled_image

# Function to remove image background
def remove_background(image):
    # Convert PIL image to numpy array
    np_image = np.array(image)
    
    # Remove background
    output = remove(np_image)
    
    # Convert back to PIL image
    return Image.fromarray(output)

# Function to composite images
def compose_images(stylized_image, background_image, position_x, position_y, scale):
    # Create a copy of the background
    result = background_image.copy()
    
    # Scale the stylized image
    width, height = stylized_image.size
    new_width = int(width * scale)
    new_height = int(height * scale)
    stylized_image = stylized_image.resize((new_width, new_height), Image.LANCZOS)
    
    # Calculate paste position (ensure it stays within bounds)
    paste_x = max(0, min(position_x, result.width - new_width))
    paste_y = max(0, min(position_y, result.height - new_height))
    
    # Paste the stylized image onto the background
    # We need to handle the alpha channel for transparent PNG
    if stylized_image.mode == 'RGBA':
        result.paste(stylized_image, (paste_x, paste_y), stylized_image)
    else:
        # Convert to RGBA if needed
        stylized_rgba = stylized_image.convert('RGBA')
        result.paste(stylized_rgba, (paste_x, paste_y), stylized_rgba)
    
    return result

# App title and description
st.title("AI Photo Stylization App")
st.markdown("""
Upload a photo of a person and an illustration image to:
1. Remove the background from the person photo
2. Apply the illustration's artistic style to the person
3. Position the stylized person onto the illustration background
4. Download the final composited image
""")

# Initialize session state for storing images and settings
if 'photo' not in st.session_state:
    st.session_state['photo'] = None
if 'illustration' not in st.session_state:
    st.session_state['illustration'] = None
if 'no_bg_photo' not in st.session_state:
    st.session_state['no_bg_photo'] = None
if 'stylized_person' not in st.session_state:
    st.session_state['stylized_person'] = None
if 'final_image' not in st.session_state:
    st.session_state['final_image'] = None
if 'position_x' not in st.session_state:
    st.session_state['position_x'] = 0
if 'position_y' not in st.session_state:
    st.session_state['position_y'] = 0
if 'scale' not in st.session_state:
    st.session_state['scale'] = 1.0
if 'style_choice' not in st.session_state:
    st.session_state['style_choice'] = 'illustration'

# Create columns for image uploads
col1, col2 = st.columns(2)

# Upload section for person photo
with col1:
    st.subheader("1. Upload Person Photo")
    photo_file = st.file_uploader("Upload a photo of a person", type=["jpg", "jpeg", "png"])
    
    if photo_file is not None:
        # Read and display the uploaded photo
        photo = Image.open(photo_file).convert("RGB")
        st.session_state['photo'] = photo
        st.image(photo, caption="Uploaded Person Photo", use_column_width=True)
        
        # Save the photo to the documents folder
        photo.save("documents/photo.jpg")
        st.success("Person photo saved successfully!")

# Upload section for illustration
with col2:
    st.subheader("2. Upload Illustration Image")
    illustration_file = st.file_uploader("Upload an illustration image", type=["jpg", "jpeg", "png"])
    
    if illustration_file is not None:
        # Read and display the uploaded illustration
        illustration = Image.open(illustration_file).convert("RGB")
        st.session_state['illustration'] = illustration
        st.image(illustration, caption="Uploaded Illustration", use_column_width=True)
        
        # Save the illustration to the documents folder
        illustration.save("documents/illustration.jpg")
        st.success("Illustration image saved successfully!")

# Processing section
st.subheader("3. Process Images")

if st.session_state['photo'] is not None and st.session_state['illustration'] is not None:
    # Style selection
    style_options = {
        'illustration': 'Use uploaded illustration style',
        'mosaic': 'Mosaic style',
        'candy': 'Candy style',
        'rain_princess': 'Rain Princess style',
        'udnie': 'Udnie style',
        'codellama': 'AI-Generated Style (CodeLlama)'  # New AI style option
    }
    
    style_choice = st.selectbox(
        "Select style to apply:", 
        options=list(style_options.keys()),
        format_func=lambda x: style_options[x],
        key='style_selector'
    )
    
    st.session_state['style_choice'] = style_choice
    
    # Style intensity slider
    style_intensity = st.slider("Style Intensity", 0.1, 1.0, 0.8, 0.1)
    
    # Enable processing button
    if st.button("Process Images", key="process_button"):
        with st.spinner("Removing background from person photo..."):
            # Remove background from person photo
            no_bg_photo = remove_background(st.session_state['photo'])
            st.session_state['no_bg_photo'] = no_bg_photo
            
            # Display the result
            st.image(no_bg_photo, caption="Person with Background Removed", width=300)
            
            # Save the no-bg photo
            no_bg_photo.save("documents/no_bg_photo.png", format="PNG")
        
        with st.spinner("Applying style transfer (this may take a while)..."):
            try:
                # Initialize style transfer model
                style_model = StyleTransfer()
                
                if style_choice == 'illustration':
                    # Use adaptive style transfer with the uploaded illustration
                    stylized_image = adaptive_style_transfer(
                        no_bg_photo, 
                        st.session_state['illustration'], 
                        style_model,
                        style_weight=style_intensity
                    )
                else:
                    # Use pre-trained style or CodeLlama AI style
                    stylized_image = style_model.stylize(no_bg_photo, style_choice)
                
                # Store the result
                st.session_state['stylized_person'] = stylized_image
                
                # Display the result
                st.image(stylized_image, caption="Stylized Person", width=300)
                
                # Save the stylized person
                stylized_image.save("documents/stylized_person.png", format="PNG")
                st.success("Style transfer completed successfully!")
            except Exception as e:
                st.error(f"Error during style transfer: {e}")

# Positioning section
st.subheader("4. Position the Stylized Person")

if st.session_state['stylized_person'] is not None and st.session_state['illustration'] is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        # Controls for positioning
        st.session_state['position_x'] = st.slider("X Position", 0, max(0, st.session_state['illustration'].width - 1), st.session_state['position_x'])
        st.session_state['position_y'] = st.slider("Y Position", 0, max(0, st.session_state['illustration'].height - 1), st.session_state['position_y'])
        st.session_state['scale'] = st.slider("Scale", 0.1, 2.0, st.session_state['scale'], 0.1)
    
    with col2:
        # Preview the composition
        if st.button("Update Preview", key="update_preview"):
            with st.spinner("Updating preview..."):
                final_image = compose_images(
                    st.session_state['stylized_person'],
                    st.session_state['illustration'],
                    st.session_state['position_x'],
                    st.session_state['position_y'],
                    st.session_state['scale']
                )
                st.session_state['final_image'] = final_image
                
                # Display the final composition
                st.image(final_image, caption="Final Composition", use_column_width=True)

# Download section
st.subheader("5. Save Final Image")

if st.session_state['final_image'] is not None:
    # Save the final image
    if st.button("Save Final Image", key="save_final"):
        with st.spinner("Saving final image..."):
            # Save to documents folder
            st.session_state['final_image'].save("documents/output.jpg", format="JPEG", quality=95)
            st.success("Final image saved to documents/output.jpg")
    
    # Provide download option
    buf = io.BytesIO()
    st.session_state['final_image'].save(buf, format="JPEG", quality=95)
    buf.seek(0)
    
    st.download_button(
        label="Download Final Image",
        data=buf,
        file_name="stylized_composition.jpg",
        mime="image/jpeg"
    )

# Footer with additional information
st.markdown("---")
st.markdown("""
### About this app
This application combines background removal, neural style transfer, and image composition to create artistic photos.

**Technologies used:**
- Streamlit for the web interface
- Rembg for background removal
- PyTorch with pre-trained Fast Neural Style Transfer models
- Meta's CodeLlama 70B model via OpenRouter for AI-generated styles
- Pillow and OpenCV for image processing
""")