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
import psutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enterprise AI Vision API", version="1.0.0")

# CORS configuration for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None
model_info = {}

# Industry-specific configurations
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
    """Load YOLOv8 model with comprehensive analysis"""
    global model, model_info
    
    try:
        logger.info("Loading YOLOv8 model...")
        model = YOLO('yolov8n.pt')
        
        # Device configuration
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        
        # Model analysis
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        model_size = sum(p.numel() * p.element_size() for p in model.model.parameters()) / (1024**2)
        
        # Performance benchmarking
        dummy_input = torch.randn(1, 3, 640, 640).to(device)
        warmup_times = []
        
        for _ in range(5):
            start = time.time()
            with torch.no_grad():
                _ = model.model(dummy_input)
            warmup_times.append(time.time() - start)
        
        model_info = {
            'architecture': 'YOLOv8n',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': round(model_size, 2),
            'device': device,
            'precision': 'FP32',
            'input_shape': [640, 640, 3],
            'num_classes': len(model.names),
            'class_names': list(model.names.values()),
            'warmup_time_avg': np.mean(warmup_times),
            'warmup_time_std': np.std(warmup_times),
            'theoretical_max_fps': round(1.0 / np.mean(warmup_times), 2),
            'memory_footprint': round(torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else psutil.Process().memory_info().rss / (1024**2), 2)
        }
        
        logger.info(f"Model loaded successfully on {device}")
        return True
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        return False

def process_frame_with_yolo(image_data: bytes, confidence_threshold: float = 0.5, industry: str = "general") -> Dict[str, Any]:
    """Process image with YOLOv8 and return detections with business intelligence"""
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV processing
        if len(image_np.shape) == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_np
            
        original_height, original_width = image_bgr.shape[:2]
        
        # Preprocessing
        preprocess_start = time.time()
        if original_width > 640:
            scale = 640 / original_width
            new_width, new_height = int(original_width * scale), int(original_height * scale)
            processed_image = cv2.resize(image_bgr, (new_width, new_height))
        else:
            processed_image = image_bgr.copy()
            scale = 1.0
            
        preprocess_time = time.time() - preprocess_start
        
        # Model inference
        inference_start = time.time()
        results = model(processed_image, conf=confidence_threshold, verbose=False)
        inference_time = time.time() - inference_start
        
        # Post-processing
        postprocess_start = time.time()
        detections = []
        annotated_image = processed_image.copy()
        
        industry_config = INDUSTRY_CONFIGS.get(industry, INDUSTRY_CONFIGS["general"])
        objects_of_interest = industry_config["objects_of_interest"]
        revenue_mapping = industry_config["revenue_per_detection"]
        
        # Enhanced color coding
        priority_colors = {
            'high': (0, 255, 0),      # Green
            'medium': (255, 255, 0),   # Yellow  
            'low': (255, 0, 0),        # Red
            'alert': (255, 0, 255)     # Magenta
        }
        
        # Process detections with intelligent filtering
        raw_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    class_name = model.names[cls]
                    
                    # Apply intelligent filtering
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    bbox_area = bbox_width * bbox_height
                    aspect_ratio = bbox_width / max(bbox_height, 1)
                    
                    # Size and aspect ratio filtering
                    frame_area = processed_image.shape[0] * processed_image.shape[1]
                    relative_size = bbox_area / frame_area
                    
                    should_include = True
                    
                    # Size filtering
                    if relative_size < 0.001 or relative_size > 0.8:
                        should_include = False
                    
                    # Class-specific filtering
                    if class_name == "person":
                        if aspect_ratio > 3 or aspect_ratio < 0.2:
                            should_include = False
                        if conf < 0.6:
                            should_include = False
                    
                    # Confidence adjustment
                    adjusted_conf = conf
                    if class_name == "person" and 0.4 <= aspect_ratio <= 1.5:
                        adjusted_conf *= 1.2
                    elif class_name in ["car", "truck"] and 1.2 <= aspect_ratio <= 4.0:
                        adjusted_conf *= 1.1
                    
                    if aspect_ratio > 5 or aspect_ratio < 0.1:
                        adjusted_conf *= 0.7
                    
                    adjusted_conf = min(adjusted_conf, 1.0)
                    
                    if adjusted_conf < confidence_threshold:
                        should_include = False
                    
                    if should_include:
                        raw_detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'conf': adjusted_conf,
                            'class': class_name,
                            'cls_id': cls,
                            'area': float(bbox_area),
                            'aspect_ratio': float(aspect_ratio)
                        })
        
        # Remove overlapping detections
        filtered_detections = []
        for i, det1 in enumerate(raw_detections):
            is_duplicate = False
            for j, det2 in enumerate(raw_detections):
                if i != j and det1['class'] == det2['class']:
                    # Calculate IoU
                    x1_int = max(det1['bbox'][0], det2['bbox'][0])
                    y1_int = max(det1['bbox'][1], det2['bbox'][1])
                    x2_int = min(det1['bbox'][2], det2['bbox'][2])
                    y2_int = min(det1['bbox'][3], det2['bbox'][3])
                    
                    if x1_int < x2_int and y1_int < y2_int:
                        intersection = (x2_int - x1_int) * (y2_int - y1_int)
                        union = det1['area'] + det2['area'] - intersection
                        iou = intersection / max(union, 1)
                        
                        if iou > 0.5 and det1['conf'] < det2['conf']:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                filtered_detections.append(det1)
        
        # Create final detections with business intelligence
        business_alerts = []
        
        for det in filtered_detections:
            class_name = det['class']
            conf = det['conf']
            x1, y1, x2, y2 = det['bbox']
            
            # Business priority classification
            revenue_value = revenue_mapping.get(class_name, 10)
            if isinstance(revenue_value, (int, float)):
                if revenue_value > 50:
                    priority = 'high'
                elif revenue_value > 25:
                    priority = 'medium'
                else:
                    priority = 'low'
            else:
                priority = 'medium'
            
            # Generate business alerts
            if class_name in objects_of_interest and conf > 0.7:
                business_alerts.append({
                    'type': 'object_of_interest',
                    'class': class_name,
                    'confidence': conf,
                    'value': revenue_value,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Draw detection on image
            color = priority_colors[priority]
            thickness = 4 if priority == 'high' else 3 if priority == 'medium' else 2
            
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            
            # Quality indicator
            quality_indicator = "●●●" if conf > 0.8 else "●●○" if conf > 0.6 else "●○○"
            
            # Label with revenue info
            if isinstance(revenue_value, str):
                label = f"{class_name}: {conf:.2f} {quality_indicator} ({revenue_value})"
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
            
            # Scale coordinates back to original image size
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
                'bbox_area': float(det['area'] * scale_x * scale_y),
                'aspect_ratio': float(det['aspect_ratio']),
                'revenue_value': revenue_value,
                'priority': priority,
                'quality_indicator': quality_indicator,
                'timestamp': datetime.now().isoformat()
            })
        
        postprocess_time = time.time() - postprocess_start
        total_time = time.time() - start_time
        
        # Convert annotated image to base64 for return
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'detections': detections,
            'annotated_image': annotated_image_b64,
            'performance': {
                'total_time': total_time,
                'inference_time': inference_time,
                'preprocess_time': preprocess_time,
                'postprocess_time': postprocess_time,
                'detections_count': len(detections),
                'raw_detections_count': len(raw_detections),
                'filtered_ratio': len(detections) / max(len(raw_detections), 1),
                'processing_fps': round(1.0 / total_time, 2) if total_time > 0 else 0
            },
            'business_alerts': business_alerts,
            'industry_config': industry_config
        }
        
    except Exception as e:
        logger.error(f"Frame processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model()
    if not success:
        logger.error("Failed to load model on startup")

@app.get("/")
async def root():
    return {"message": "Enterprise AI Vision API", "status": "active", "model_loaded": model is not None}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def get_model_info():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return model_info

@app.get("/industries")
async def get_industries():
    return {
        "industries": {k: v for k, v in INDUSTRY_CONFIGS.items()},
        "default": "general"
    }

@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    confidence: float = 0.5,
    industry: str = "general"
):
    """Process uploaded image and return detections"""
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Process with YOLOv8
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

@app.post("/detect/base64")
async def detect_objects_base64(
    request: Dict[str, Any]
):
    """Process base64 image and return detections"""
    
    try:
        # Extract parameters
        image_b64 = request.get("image")
        confidence = request.get("confidence", 0.5)
        industry = request.get("industry", "general")
        
        if not image_b64:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(image_b64.split(',')[1] if ',' in image_b64 else image_b64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
        
        # Process with YOLOv8
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)