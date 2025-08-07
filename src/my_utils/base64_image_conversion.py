import base64
from PIL import Image
from io import BytesIO

def image_to_base64(image):
    # Convert PIL Image to base64 encoding
    with BytesIO() as buffer:
        image.save(buffer, 'png')
        return base64.b64encode(buffer.getvalue()).decode()

def base64_to_image(base64_string):
    # Convert base64 encoding to PIL Image

    # Decode the base64 string to bytes
    image_data = base64.b64decode(base64_string)

    # Create a BytesIO object and load the image data
    image_buffer = BytesIO(image_data)

    # Open the image using PIL
    image = Image.open(image_buffer)

    return image