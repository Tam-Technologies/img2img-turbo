import requests
import subprocess

from PIL import Image

PREDICT_URL = 'https://cyclegan-turbo-inference-437403722141.us-central1.run.app/predict'

def image_to_base64(image):
    # Convert PIL Image to base64 encoding
    from io import BytesIO
    import base64
    with BytesIO() as buffer:
        image.save(buffer, 'png')
        return base64.b64encode(buffer.getvalue()).decode()

def base64_to_image(base64_string):
    # Convert base64 encoding to PIL Image
    from io import BytesIO
    import base64

    # Decode the base64 string to bytes
    image_data = base64.b64decode(base64_string)

    # Create a BytesIO object and load the image data
    image_buffer = BytesIO(image_data)

    # Open the image using PIL
    image = Image.open(image_buffer)

    return image

if __name__ == "__main__":
    img = Image.open('/Users/mtam/Documents/scans/cyclegan_turbo_results/pix2pix_turbo_segment32bit_binary_silhouette_to_scan_image/amAoyZfQ4G3oQortYD7K_silhouette0.png')
    base64_string = image_to_base64(img)
    token = subprocess.check_output(['gcloud', 'auth', 'print-identity-token']).decode().strip()
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    r = requests.post(PREDICT_URL, headers=headers,
                      json={"input_image": base64_string,
                            "prompt": "A woman in a sports bra and shorts standing in an indoor gym with arms raised, facing towards the camera",
                            "use_fp16": True})
    print(r.json()['execution_time'])