import logging
import time
import base64
import io

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from PIL import Image
import torch
from torchvision import transforms

from src.pix2pix_turbo import Pix2Pix_Turbo

# Initialize logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

class SingleImagePayload(BaseModel):
    input_image: str
    prompt: str
    pretrained_model_name: str = 'pix2pix_turbo_segment32bit_binary_silhouette_to_scan_image'
    use_fp16: bool = False

@app.post("/predict")
async def predict(payload: SingleImagePayload):
    try:
        logging.info("Received request to run inference on a single image.")
        start_time = time.time()

        # Strip off the base64 image header if it exists
        base64_image = payload.input_image
        if base64_image.startswith("data:image"):
            base64_image = base64_image.split(",")[1]
        if not base64_image:
            raise HTTPException(status_code=400, detail="Invalid image format")

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

            output_buffer = io.BytesIO()
            output_pil.save(output_buffer, format="PNG")
            output_image_base64 = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
        logging.info(f"Image segmentation processing is complete.")

        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Execution time for processing image: {execution_time} seconds")

        return {
            "message": "Image processed successfully",
            "execution_time": execution_time,
            "ouput_image": output_image_base64
        }

    except Exception as e:
        logging.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)