import requests
import subprocess

import pandas as pd
from PIL import Image
from my_utils.base64_image_conversion import image_to_base64, base64_to_image
import constants

PREDICT_URL = 'https://cyclegan-turbo-inference-437403722141.us-central1.run.app/predict'
PREDICT_PATH_URL = 'https://cyclegan-turbo-inference-437403722141.us-central1.run.app/predict-image-path'

if __name__ == "__main__":
    df = pd.read_csv('/Users/mtam/Documents/scans/synthetic_scans_dataset_csv/mixed_distribution_38k_no_ttf_shapes_no_cyclegan/train_dataset_female.csv')


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

    r = requests.post(PREDICT_PATH_URL, headers=headers,
                      json={"input_image_path": f'/gcs/synthetic_scans/001MroKY4Rkdzg0eTXvN/body/silhouettes/silhouette0.png',
                            "prompt": "A woman in a sports bra and shorts standing in an indoor gym with arms raised, facing towards the camera",
                            "use_fp16": True,
                            "output_image_path": f'/gcs/synthetic_scans/001MroKY4Rkdzg0eTXvN/body/cyclegan_turbo_images_v{constants.CYCLEGAN_TURBO_VERSION}/snapshot0.png'})
    print(r.json()['execution_time'])