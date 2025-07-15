from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime
import time
import torch
import json
from typing import List, Dict, Any
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enterprise AI Vision API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None
model_info = {}

# Industry configurations
INDUSTRY_CONFIGS = {
    "retail": {
        "name": "Retail Analytics",
        "objects_of_interest": ["person", "handbag", "backpack", "suitcase", "cell phone"],
        "revenue_per_detection": {
            "person": "visitor analytics", 
            "handbag": 15, 
            "backpack": 10, 
            "cell phone": 8
        },
        "alert_thresholds": {"crowd_density": 10, "queue_length": 5},
        "primary_color": "#10b981",
        "gradient": "from-emerald-500 to-teal-600"
    },
    "security": {
        "name": "Security & Surveillance",
        "objects_of_interest": ["person", "car", "truck", "backpack", "handbag", "knife"],
        "revenue_per_detection": {
            "person": "security monitoring", 
            "car": 75, 
            "truck": 100, 
            "backpack": 25
        },
        "alert_thresholds": {"unauthorized_person": 1, "vehicle_in_restricted": 1},
        "primary_color": "#ef4444",
        "gradient": "from-red-500 to-rose-600"
    },
    "manufacturing": {
        "name": "Manufacturing QC",
        "objects_of_interest": ["bottle", "cup", "scissors", "knife", "spoon"],
        "revenue_per_detection": {
            "bottle": 50, 
            "cup": 30, 
            "scissors": 40, 
            "knife": 45, 
            "spoon": 25
        },
        "alert_thresholds": {"defect_rate": 0.05, "throughput_drop": 0.2},
        "primary_color": "#f59e0b",
        "gradient": "from-amber-500 to-orange-600"
    },
    "general": {
        "name": "General Purpose",
        "objects_of_interest": ["person", "car", "truck", "bus", "bottle", "cup", "laptop", "cell phone"],
        "revenue_per_detection": {
            "person": "analytics", 
            "car": 30, 
            "truck": 45, 
            "bus": 60, 
            "bottle": 15, 
            "cup": 10, 
            "laptop": 85, 
            "cell phone": 25
        },
        "alert_thresholds": {"general_threshold": 5},
        "primary_color": "#3b82f6",
        "gradient": "from-blue-500 to-indigo-600"
    }
}

def load_model():
    """Load YOLOv8 model"""
    global model, model_info
    
    try:
        logger.info("Loading YOLOv8 model...")
        model = YOLO('yolov8n.pt')
        
        # Device configuration (CPU for Vercel)
        device = 'cpu'  # Vercel doesn't have GPU
        
        model_info = {
            'architecture': 'YOLOv8n',
            'device': device,
            'precision': 'FP32',
            'input_shape': [640, 640, 3],
            'num_classes': len(model.names),
            'class_names': list(model.names.values()),
            'theoretical_max_fps': 10,  # Conservative for CPU
            'model_size_mb': 6.2
        }
        
        logger.info(f"Model loaded successfully on {device}")
        return True
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        return False

def process_frame_with_yolo(image_data: bytes, confidence_threshold: float = 0.5, industry: str = "general") -> Dict[str, Any]:
    """Process image with YOLOv8 (optimized for Vercel)"""
    
    if model is None:
        # Try to load model if not loaded
        if not load_model():
            raise HTTPException(status_code=500, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_np
            
        original_height, original_width = image_bgr.shape[:2]
        
        # Resize for faster processing on Vercel
        max_size = 416  # Smaller for serverless
        if max(original_width, original_height) > max_size:
            scale = max_size / max(original_width, original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            processed_image = cv2.resize(image_bgr, (new_width, new_height))
        else:
            processed_image = image_bgr.copy()
            scale = 1.0
            
        # Model inference
        inference_start = time.time()
        results = model(processed_image, conf=confidence_threshold, verbose=False)
        inference_time = time.time() - inference_start
        
        # Process results
        detections = []
        annotated_image = processed_image.copy()
        
        industry_config = INDUSTRY_CONFIGS.get(industry, INDUSTRY_CONFIGS["general"])
        objects_of_interest = industry_config["objects_of_interest"]
        revenue_mapping = industry_config["revenue_per_detection"]
        
        priority_colors = {
            'high': (0, 255, 0),
            'medium': (255, 255, 0),
            'low': (255, 0, 0),
            'alert': (255, 0, 255)
        }
        
        business_alerts = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    class_name = model.names[cls]
                    
                    # Business priority
                    revenue_value = revenue_mapping.get(class_name, 10)
                    if isinstance(revenue_value, (int, float)):
                        priority = 'high' if revenue_value > 50 else 'medium' if revenue_value > 25 else 'low'
                    else:
                        priority = 'medium'
                    
                    # Business alerts
                    if class_name in objects_of_interest and conf > 0.7:
                        business_alerts.append({
                            'type': 'object_of_interest',
                            'class': class_name,
                            'confidence': conf,
                            'value': revenue_value,
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    # Draw detection
                    color = priority_colors[priority]
                    thickness = 3 if priority == 'high' else 2
                    
                    cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    
                    # Label
                    quality_indicator = "●●●" if conf > 0.8 else "●●○" if conf > 0.6 else "●○○"
                    
                    if isinstance(revenue_value, str):
                        label = f"{class_name}: {conf:.2f} {quality_indicator}"
                    else:
                        label = f"{class_name}: {conf:.2f} {quality_indicator} (${revenue_value})"
                    
                    # Draw label
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(annotated_image,
                                 (int(x1), int(y1 - label_size[1] - 10)),
                                 (int(x1 + label_size[0] + 10), int(y1)),
                                 color, -1)
                    
                    cv2.putText(annotated_image, label,
                               (int(x1 + 5), int(y1 - 5)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Scale back to original size
                    scale_x = original_width / processed_image.shape[1]
                    scale_y = original_height / processed_image.shape[0]
                    
                    detections.append({
                        'class': class_name,
                        'confidence': float(conf),
                        'bbox': [
                            int(x1 * scale_x), 
                            int(y1 * scale_y), 
                            int(x2 * scale_x), 
                            int(y2 * scale_y)
                        ],
                        'revenue_value': revenue_value,
                        'priority': priority,
                        'quality_indicator': quality_indicator,
                        'timestamp': datetime.now().isoformat()
                    })
        
        total_time = time.time() - start_time
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'detections': detections,
            'annotated_image': annotated_image_b64,
            'performance': {
                'total_time': total_time,
                'inference_time': inference_time,
                'detections_count': len(detections),
                'processing_fps': round(1.0 / total_time, 2) if total_time > 0 else 0
            },
            'business_alerts': business_alerts,
            'industry_config': industry_config
        }
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/model/info")
async def get_model_info():
    if model is None:
        load_model()
    return model_info

@app.get("/api/industries")
async def get_industries():
    return {
        "industries": INDUSTRY_CONFIGS,
        "default": "general"
    }

@app.post("/api/detect")
async def detect_objects(
    file: UploadFile = File(...),
    confidence: float = 0.5,
    industry: str = "general"
):
    """Process uploaded image"""
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        result = process_frame_with_yolo(image_data, confidence, industry)
        
        return {
            "success": True,
            "filename": file.filename,
            "confidence_threshold": confidence,
            "industry": industry,
            **result
        }
        
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect/base64")
async def detect_objects_base64(request: Dict[str, Any]):
    """Process base64 image"""
    
    try:
        image_b64 = request.get("image")
        confidence = request.get("confidence", 0.5)
        industry = request.get("industry", "general")
        
        if not image_b64:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # Decode base64
        try:
            image_data = base64.b64decode(image_b64.split(',')[1] if ',' in image_b64 else image_b64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
        
        result = process_frame_with_yolo(image_data, confidence, industry)
        
        return {
            "success": True,
            "confidence_threshold": confidence,
            "industry": industry,
            **result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Base64 detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

# Root endpoint
@app.get("/api")
async def root():
    return {"message": "Enterprise AI Vision API", "status": "active", "model_loaded": model is not None}