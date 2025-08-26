import logging
import time
import base64
import io
import os

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from PIL import Image
import torch
from torchvision import transforms

from src.pix2pix_turbo import Pix2Pix_Turbo
from src.my_utils.base64_image_conversion import image_to_base64, base64_to_image
from src import firebase_utils

# Initialize logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Global model instance - load once at startup
model = None

# Get environment variables set by Vertex AI
AIP_HTTP_PORT = int(os.getenv('AIP_HTTP_PORT', '8080'))
AIP_HEALTH_ROUTE = os.getenv('AIP_HEALTH_ROUTE', '/health')
AIP_PREDICT_ROUTE = os.getenv('AIP_PREDICT_ROUTE', '/predict')

logging.info(f"Server will run on port: {AIP_HTTP_PORT}")
logging.info(f"Health route: {AIP_HEALTH_ROUTE}")
logging.info(f"Predict route: {AIP_PREDICT_ROUTE}")


# Vertex AI request/response models
class VertexAIRequest(BaseModel):
    instances: List[Dict[str, Any]]
    parameters: Optional[Dict[str, Any]] = None

class VertexAIResponse(BaseModel):
    predictions: List[Dict[str, Any]]

# Legacy models for backward compatibility
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
    output_segment32bit_path: str = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup to avoid loading on each request"""
    global model
    try:
        logging.info("Loading model on startup...")
        # Use default model name, can be overridden by parameters
        model = Pix2Pix_Turbo(pretrained_name='pix2pix_turbo_segment32bit_binary_silhouette_to_scan_image')
        model.set_eval()
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model on startup: {e}")
        # Don't fail startup, allow model loading on first request
        model = None

# Vertex AI health check endpoint
@app.get(AIP_HEALTH_ROUTE)
async def health_check():
    """Health check endpoint for Vertex AI"""
    global model
    if model is None:
        # Try to load model if not loaded
        try:
            model = Pix2Pix_Turbo(pretrained_name='pix2pix_turbo_segment32bit_binary_silhouette_to_scan_image')
            model.set_eval()
            logging.info("Model loaded during health check")
        except Exception as e:
            logging.error(f"Model not ready: {e}")
            raise HTTPException(status_code=503, detail="Model not ready")
    
    return {"status": "healthy", "model_loaded": model is not None}

# Vertex AI prediction endpoint
@app.post(AIP_PREDICT_ROUTE)
async def vertex_ai_predict(request: VertexAIRequest):
    """Vertex AI compatible prediction endpoint with file path functionality"""
    global model
    
    try:
        logging.info("Received Vertex AI prediction request")
        
        if model is None:
            # Load model if not already loaded
            model = Pix2Pix_Turbo(pretrained_name='pix2pix_turbo_segment32bit_binary_silhouette_to_scan_image')
            model.set_eval()
            logging.info("Model loaded for prediction")
        
        predictions = []
        
        # Process each instance
        for instance in request.instances:
            # Extract required fields from instance
            input_image_path = instance.get('input_image_path')
            prompt = instance.get('prompt', '')
            
            if not input_image_path:
                raise HTTPException(status_code=400, detail="input_image_path is required in each instance")
            
            # Get parameters (use instance-level params or global params)
            params = request.parameters or {}
            instance_params = instance.get('parameters', {})
            params.update(instance_params)
            
            pretrained_model_name = params.get('pretrained_model_name', 'pix2pix_turbo_segment32bit_binary_silhouette_to_scan_image')
            use_fp16 = params.get('use_fp16', False)
            output_image_path = instance.get('output_image_path')
            output_segment32bit_path = instance.get('output_segment32bit_path')
            
            # Process the image using file path functionality
            prediction = await process_image_path(
                input_image_path=input_image_path,
                prompt=prompt,
                pretrained_model_name=pretrained_model_name,
                use_fp16=use_fp16,
                output_image_path=output_image_path,
                output_segment32bit_path=output_segment32bit_path
            )
            
            predictions.append(prediction)
        
        return VertexAIResponse(predictions=predictions)
    
    except HTTPException as e:
        logging.error(f"HTTP error in Vertex AI predict: {e.detail}")
        raise e
    except Exception as e:
        logging.error(f"Error in Vertex AI predict: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def process_image_path(input_image_path: str, prompt: str, pretrained_model_name: str, use_fp16: bool = False, output_image_path: str = None, output_segment32bit_path: str = None):
    """Process image from file path and save to output path - matches predict-image-path functionality"""
    global model
    
    try:
        logging.info(f"Received request to run inference on image at path {input_image_path}")
        
        # Validate input file exists
        if not os.path.exists(input_image_path):
            raise HTTPException(status_code=299, detail=f"Image not found at {input_image_path}")
        
        # Validate output file doesn't exist
        if output_image_path and os.path.exists(output_image_path):
            raise HTTPException(status_code=299, detail=f"Output image already exists at {output_image_path}")
        
        # Load and convert image to base64
        img = Image.open(input_image_path)
        base64_string = image_to_base64(img)
        
        # Create payload for processing
        single_image_payload = SingleImagePayload(
            input_image=base64_string,
            prompt=prompt,
            pretrained_model_name=pretrained_model_name,
            use_fp16=use_fp16
        )
        
        # Process the image using legacy predict function
        result = await predict(single_image_payload)
        
        # Save output image if path specified
        if output_image_path:
            logging.info(f"Saving output image to {output_image_path}")
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            output_pil = base64_to_image(result['output_image'])
            output_pil.save(output_image_path)
        
        # Create segmentation task if path specified
        if output_segment32bit_path:
            logging.info(f"Creating segmentation task for {output_segment32bit_path}")
            firebase_utils.create_segment_image_task(
                result['output_image'], 
                use_32bit=True, 
                preprocess=True,
                output_image_path=output_segment32bit_path
            )
        
        # Return result with additional path information
        return {
            "output_image": result['output_image'],
            "execution_time": result['execution_time'],
            "prompt": prompt,
            "input_image_path": input_image_path,
            "output_image_path": output_image_path,
            "output_segment32bit_path": output_segment32bit_path,
            "message": "Image processed successfully"
        }
        
    except HTTPException as e:
        logging.error(f"Error processing image path: {e.detail}", exc_info=True)
        raise e
    except Exception as e:
        logging.error(f"Error processing image path: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def process_single_image(input_image: str, prompt: str, pretrained_model_name: str, use_fp16: bool = False):
    """Process a single image and return prediction - for base64 input"""
    global model
    
    start_time = time.time()
    
    # Strip off the base64 image header if it exists
    base64_image = input_image
    if base64_image.startswith("data:image"):
        base64_image = base64_image.split(",")[1]
    if not base64_image:
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    if torch.cuda.is_available():
        logging.info("Using CUDA")
    else:
        logging.info("Using CPU")
    
    # Use global model or create new one if different pretrained_model_name
    current_model = model
    if model is None or (hasattr(model, 'pretrained_name') and model.pretrained_name != pretrained_model_name):
        current_model = Pix2Pix_Turbo(pretrained_name=pretrained_model_name)
        current_model.set_eval()
        logging.info(f"Loaded model: {pretrained_model_name}")
    
    if use_fp16:
        current_model.half()
    
    # Process image
    image_data = base64.b64decode(base64_image)
    input_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
    new_width = input_pil.width - input_pil.width % 8
    new_height = input_pil.height - input_pil.height % 8
    input_pil = input_pil.resize((new_width, new_height), Image.LANCZOS)
    
    # Generate image
    with torch.no_grad():
        c_t = transforms.ToTensor()(input_pil).unsqueeze(0)
        if torch.cuda.is_available():
            c_t = c_t.cuda()
        if use_fp16:
            c_t = c_t.half()
        output_image = current_model(c_t, prompt)
        
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        output_image_base64 = image_to_base64(output_pil)
    
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Image processing completed in {execution_time:.2f} seconds")
    
    return {
        "output_image": output_image_base64,
        "execution_time": execution_time,
        "prompt": prompt
    }

@app.post("/predict-image-path")
async def predict_path(payload: SingleImagePathPayload):
    try:
        logging.info(f"Received request to run inference on a single image at path {payload.input_image_path}.")
        if not os.path.exists(payload.input_image_path):
            raise HTTPException(status_code=299, detail=f"Image not found at {payload.input_image_path}")
        if payload.output_image_path and os.path.exists(payload.output_image_path):
            raise HTTPException(status_code=299,
                              detail=f"Output image already exists at {payload.output_image_path}")


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

        if payload.output_segment32bit_path:
            firebase_utils.create_segment_image_task(result['output_image'], use_32bit=True, preprocess=True,
                                                     output_image_path=payload.output_segment32bit_path)

        return result

    except HTTPException as e:
        logging.error(f"Error processing request: {e.detail}", exc_info=True)
        raise e

    except Exception as e:
        logging.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Legacy endpoint for backward compatibility
@app.post("/predict")
async def predict(payload: SingleImagePayload):
    """Legacy prediction endpoint for backward compatibility"""
    try:
        logging.info('Received request to run inference on a single image')
        
        prediction = await process_single_image(
            input_image=payload.input_image,
            prompt=payload.prompt,
            pretrained_model_name=payload.pretrained_model_name,
            use_fp16=payload.use_fp16
        )
        
        return {
            "message": "Image processed successfully",
            "execution_time": prediction["execution_time"],
            "output_image": prediction["output_image"]
        }
        
    except HTTPException as e:
        logging.error(f"Error processing request: {e.detail}", exc_info=True)
        raise e
    except Exception as e:
        logging.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=AIP_HTTP_PORT)