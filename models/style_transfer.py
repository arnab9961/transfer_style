import tensorflow as tf
import numpy as np
from PIL import Image
import os

class StyleTransfer:
    def __init__(self):
        """Initialize the StyleTransfer class."""
        # Download and cache the model files
        print('Downloading the model files...')
        self.style_predict_path = tf.keras.utils.get_file(
            'style_predict.tflite', 
            'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/prediction/1?lite-format=tflite'
        )
        self.style_transform_path = tf.keras.utils.get_file(
            'style_transform.tflite', 
            'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/transfer/1?lite-format=tflite'
        )
        print('Model files downloaded')
    
    def load_content_img(self, image):
        """Load and preprocess content image from a numpy array."""
        img = tf.convert_to_tensor(image)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img[tf.newaxis, :]
        return img
    
    def load_style_img(self, image_path):
        """Load and preprocess style image from a file path."""
        img = tf.io.read_file(image_path)
        img = tf.io.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img[tf.newaxis, :]
        return img
    
    def preprocess_image(self, image, target_dim):
        """Resize and crop image to target dimensions."""
        shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
        short_dim = min(shape)
        scale = target_dim / short_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        
        image = tf.image.resize(image, new_shape)
        image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)
        
        return image
    
    def run_style_predict(self, preprocessed_image):
        """Run the style prediction model on the image."""
        interpreter = tf.lite.Interpreter(model_path=self.style_predict_path)
        
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        interpreter.set_tensor(input_details[0]["index"], preprocessed_image)
        
        interpreter.invoke()
        style_bottleneck = interpreter.tensor(
            interpreter.get_output_details()[0]["index"]
        )()
        
        return style_bottleneck
    
    def run_style_transform(self, style_bottleneck, preprocessed_content_image):
        """Apply style bottleneck to the content image."""
        interpreter = tf.lite.Interpreter(model_path=self.style_transform_path)
        input_details = interpreter.get_input_details()
        interpreter.allocate_tensors()
        
        interpreter.set_tensor(input_details[0]["index"], preprocessed_content_image)
        interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
        interpreter.invoke()
        
        stylized_image = interpreter.tensor(
            interpreter.get_output_details()[0]["index"]
        )()
        
        return stylized_image
    
    def transfer_style(self, content_image, style_image_path, content_blending_ratio=0.5):
        """Apply style transfer from style image to content image.
        
        Args:
            content_image: PIL Image or numpy array of the content image
            style_image_path: File path to the style image
            content_blending_ratio: Float between 0 and 1, higher values preserve more content
            
        Returns:
            PIL Image of the stylized result
        """
        # Convert PIL Image to numpy array if necessary
        if not isinstance(content_image, np.ndarray):
            content_image = np.array(content_image)
        
        # Prepare content image
        content_tensor = self.load_content_img(content_image)
        preprocessed_content = self.preprocess_image(content_tensor, 384)
        
        # Prepare style image
        style_tensor = self.load_style_img(style_image_path)
        preprocessed_style = self.preprocess_image(style_tensor, 256)
        
        # Calculate style bottleneck
        style_bottleneck = self.run_style_predict(preprocessed_style)
        
        # Calculate content bottleneck
        content_bottleneck = self.run_style_predict(
            self.preprocess_image(content_tensor, 256)
        )
        
        # Blend bottlenecks based on the ratio
        style_bottleneck_blended = content_blending_ratio * content_bottleneck + (1 - content_blending_ratio) * style_bottleneck
        
        # Run style transfer
        stylized_image = self.run_style_transform(style_bottleneck_blended, preprocessed_content)
        
        # Convert to displayable format
        result_image = tf.squeeze(stylized_image, axis=0)
        result_pil = Image.fromarray(np.array(result_image * 255, dtype=np.uint8))
        
        return result_pil