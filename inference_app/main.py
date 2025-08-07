import logging
import time
import base64
import io
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from PIL import Image
import torch
from torchvision import transforms

from src.pix2pix_turbo import Pix2Pix_Turbo
from src.my_utils.base64_image_conversion import image_to_base64, base64_to_image

# Initialize logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

class SingleImagePayload(BaseModel):
    input_image: str
    prompt: str
    pretrained_model_name: str = 'pix2pix_turbo_segment32bit_binary_silhouette_to_scan_image'
    use_fp16: bool = False

class SingleImagePathPayload(BaseModel):
    input_image_path: str
    prompt: str
    pretrained_model_name: str = 'pix2pix_turbo_segment32bit_binary_silhouette_to_scan_image'
    use_fp16: bool = False
    output_image_path: str = None

@app.post("/predict-image-path")
async def predict_path(payload: SingleImagePathPayload):
    logging.info(f"Received request to run inference on a single image at path {payload.input_image_path}.")
    if not os.path.exists(payload.input_image_path):
        raise HTTPException(status_code=404, detail=f"Image not found at {payload.input_image_path}")
    if payload.output_image_path and os.path.exists(payload.output_image_path):
        raise HTTPException(status_code=400, detail=f"Output image already exists at {payload.output_image_path}")

    try:
        img = Image.open(payload.input_image_path)
        base64_string = image_to_base64(img)

        single_image_payload = SingleImagePayload(
            input_image=base64_string,
            prompt=payload.prompt,
            pretrained_model_name=payload.pretrained_model_name,
            use_fp16=payload.use_fp16
        )
        result = await predict(single_image_payload)

        if payload.output_image_path:
            logging.info(f"Saving output image to {payload.output_image_path}")
            os.makedirs(os.path.dirname(payload.output_image_path), exist_ok=True)
            output_pil = base64_to_image(result['output_image'])
            output_pil.save(payload.output_image_path)
        return result

    except Exception as e:
        logging.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(payload: SingleImagePayload):
    logging.info("Received request to run inference on a single image.")
    start_time = time.time()

    # Strip off the base64 image header if it exists
    base64_image = payload.input_image
    if base64_image.startswith("data:image"):
        base64_image = base64_image.split(",")[1]
    if not base64_image:
        raise HTTPException(status_code=400, detail="Invalid image format")

    try:
        if torch.cuda.is_available():
            print("Using CUDA")
        else:
            print("Using CPU")

        # Initialize the model
        model = Pix2Pix_Turbo(pretrained_name=payload.pretrained_model_name)
        model.set_eval()
        if payload.use_fp16:
            model.half()

        # make sure that the input image is a multiple of 8
        image_data = base64.b64decode(base64_image)
        input_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)

        # translate the image
        with torch.no_grad():
            c_t = transforms.ToTensor()(input_image).unsqueeze(0)
            if torch.cuda.is_available():
                c_t = c_t.cuda()
            if payload.use_fp16:
                c_t = c_t.half()
            output_image = model(c_t, payload.prompt)

            output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)

            output_image_base64 = image_to_base64(output_pil)
        logging.info(f"Image processing is complete.")

        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Execution time for processing image: {execution_time} seconds")

        return {
            "message": "Image processed successfully",
            "execution_time": execution_time,
            "output_image": output_image_base64
        }

    except Exception as e:
        logging.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)