import logging
import time
import base64
import io
import os
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from PIL import Image
import torch
from torchvision import transforms
from google.cloud import storage

from src.pix2pix_turbo import Pix2Pix_Turbo
from src.my_utils.base64_image_conversion import image_to_base64, base64_to_image
from src import firebase_utils
from src import constants

# Initialize logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Global model instance - load once at startup
model = None

# GCS client - initialize once
gcs_client = None

def initialize_gcs_client():
    """Initialize GCS client"""
    global gcs_client
    if gcs_client is None:
        gcs_client = storage.Client()
        logging.info("GCS client initialized successfully")

def is_gcs_path(path: str) -> bool:
    """Check if path is a GCS path"""
    return path.startswith('gs://') or path.startswith('/gcs')

def parse_gcs_path(path: str) -> tuple:
    """Parse GCS path into bucket and blob name"""
    if path.startswith('gs://'):
        parsed = urlparse(path)
        bucket_name = parsed.netloc
        blob_name = parsed.path.lstrip('/')
        return bucket_name, blob_name
    elif path.startswith('/gcs'):
        # Remove /gcs prefix and use default bucket
        blob_name = path[4:].lstrip('/')
        return constants.VERTEX_AI_BUCKET_NAME, blob_name
    return None, None

def download_from_gcs(gcs_path: str) -> bytes:
    """Download file from GCS and return bytes"""
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    if not bucket_name or not blob_name:
        raise HTTPException(status_code=400, detail=f"Invalid GCS path: {gcs_path}")
    
    try:
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        if not blob.exists():
            raise HTTPException(status_code=404, detail=f"File not found in GCS: {gcs_path}")
        
        return blob.download_as_bytes()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download from GCS: {str(e)}")

def upload_to_gcs(gcs_path: str, image: Image.Image) -> None:
    """Upload PIL Image to GCS"""
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    if not bucket_name or not blob_name:
        raise HTTPException(status_code=400, detail=f"Invalid GCS path: {gcs_path}")
    
    try:
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        # Determine format from file extension or default to PNG
        format = 'PNG'
        if blob_name.lower().endswith('.jpg') or blob_name.lower().endswith('.jpeg'):
            format = 'JPEG'
        elif blob_name.lower().endswith('.webp'):
            format = 'WEBP'
        
        image.save(img_byte_arr, format=format)
        img_byte_arr.seek(0)
        
        # Upload to GCS
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_file(img_byte_arr, content_type=f'image/{format.lower()}')
        
        logging.info(f"Successfully uploaded image to GCS: {gcs_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload to GCS: {str(e)}")

def save_image_with_gcs_support(image: Image.Image, output_path: str) -> None:
    """Save image to local filesystem or GCS based on path"""
    if is_gcs_path(output_path):
        # Initialize GCS client if needed
        if gcs_client is None:
            initialize_gcs_client()
        upload_to_gcs(output_path, image)
    else:
        # Save to local filesystem
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)

def file_exists_with_fallback(file_path: str) -> bool:
    """Check if file exists locally or in GCS"""
    # First, check local filesystem
    if os.path.exists(file_path):
        return True
    
    # If path is GCS, check GCS
    if is_gcs_path(file_path):
        if gcs_client is None:
            initialize_gcs_client()
        
        bucket_name, blob_name = parse_gcs_path(file_path)
        if bucket_name and blob_name:
            try:
                bucket = gcs_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                return blob.exists()
            except Exception:
                return False
    
    # Try GCS with default bucket as fallback
    gcs_path = f"gs://{constants.VERTEX_AI_BUCKET_NAME}/{file_path.lstrip('/')}"
    try:
        if gcs_client is None:
            initialize_gcs_client()
        bucket = gcs_client.bucket(constants.VERTEX_AI_BUCKET_NAME)
        blob = bucket.blob(file_path.lstrip('/'))
        return blob.exists()
    except Exception:
        return False

def load_image_with_fallback(image_path: str) -> Image.Image:
    """Load image from local filesystem or GCS with fallback logic"""
    # First, try to load from local filesystem
    if os.path.exists(image_path):
        return Image.open(image_path).convert('RGB')
    
    # If not found locally and path is GCS, try GCS
    if is_gcs_path(image_path):
        if gcs_client is None:
            initialize_gcs_client()
        
        image_bytes = download_from_gcs(image_path)
        return Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Try GCS with default bucket as fallback
    gcs_path = f"gs://{constants.VERTEX_AI_BUCKET_NAME}/{image_path.lstrip('/')}"
    try:
        if gcs_client is None:
            initialize_gcs_client()
        image_bytes = download_from_gcs(gcs_path)
        return Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except HTTPException:
        pass  # Continue to raise the original error
    
    # If all else fails, raise file not found error
    raise HTTPException(status_code=299, detail=f"Image not found at {image_path}")

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
    
    # Initialize GCS client
    initialize_gcs_client()

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
        
        # Check if output path already exists, skip processing if it does
        if output_image_path and file_exists_with_fallback(output_image_path):
            return {"message": f"Output image already exists at {output_image_path}, skipping inference"}
        
        # Load and convert image to base64
        img = load_image_with_fallback(input_image_path)
        base64_string = image_to_base64(img)
        
        # Create payload for processing
        single_image_payload = SingleImagePayload(
            input_image=base64_string,
            prompt=prompt,
            pretrained_model_name=pretrained_model_name,
            use_fp16=use_fp16
        )
        
        # Process the image using legacy predict function
        logging.info(f"Processing image {input_image_path} with prompt {prompt}")
        result = await predict(single_image_payload)
        
        # Save output image if path specified
        if output_image_path:
            logging.info(f"Saving output image to {output_image_path}")
            output_pil = base64_to_image(result['output_image'])
            save_image_with_gcs_support(output_pil, output_image_path)
        
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
        if payload.output_image_path and file_exists_with_fallback(payload.output_image_path):
            return {"message": f"Output image already exists at {payload.output_image_path}, skipping inference"}

        img = load_image_with_fallback(payload.input_image_path)
        base64_string = image_to_base64(img)

        logging.info(f"Image loaded successfully from {payload.input_image_path}")

        single_image_payload = SingleImagePayload(
            input_image=base64_string,
            prompt=payload.prompt,
            pretrained_model_name=payload.pretrained_model_name,
            use_fp16=payload.use_fp16
        )
        result = await predict(single_image_payload)

        if payload.output_image_path:
            logging.info(f"Saving output image to {payload.output_image_path}")
            output_pil = base64_to_image(result['output_image'])
            save_image_with_gcs_support(output_pil, payload.output_image_path)

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