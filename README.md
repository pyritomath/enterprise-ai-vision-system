# Enterprise AI Vision System

A professional computer vision platform combining **Next.js frontend** with **FastAPI + YOLOv8 backend** for enterprise-grade object detection with business intelligence.

## 🚀 Features

- **Real YOLOv8 Object Detection** - Powered by Ultralytics YOLOv8
- **Industry-Specific Configurations** - Retail, Security, Manufacturing, General
- **Business Intelligence** - ROI calculations, revenue tracking, performance metrics
- **Real-time Processing** - Live camera feed with instant detection
- **Dark/Light Mode** - Professional UI with smooth transitions
- **Performance Analytics** - Comprehensive metrics and monitoring
- **Enterprise Ready** - Production-ready architecture

## 🏗️ Architecture

```
Frontend (Next.js)     →     Backend (FastAPI)
├── Camera Access           ├── YOLOv8 Model
├── Dark/Light Mode         ├── Object Detection
├── Business Intelligence   ├── Image Processing
├── Real-time Stats        ├── Business Logic
└── Industry Config        └── Performance Metrics
```

## 🛠️ Setup & Installation

### Backend Setup (Python FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend Setup (Next.js)

```bash
npm install
npm run dev
```

## 🚀 Deployment

### Backend Deployment (Railway/Render):
1. Connect GitHub repo
2. Auto-deploy on push
3. Set environment variables

### Frontend Deployment (Vercel):
```bash
npm run build
vercel --prod
```

Set environment variable:
```
NEXT_PUBLIC_API_URL=https://your-backend-url.com
```

## 📊 Industry Configurations

- **Retail Analytics** - Customer behavior, foot traffic
- **Security & Surveillance** - Intrusion detection, threat assessment
- **Manufacturing QC** - Quality control, defect detection
- **General Purpose** - Multi-purpose object detection

## 🎨 Tech Stack

**Frontend:** Next.js 14, React 18, Tailwind CSS
**Backend:** FastAPI, YOLOv8, OpenCV, PyTorch
**Deployment:** Vercel + Railway/Render

## 📱 Usage

1. **Start Backend**: `uvicorn main:app --reload`
2. **Start Frontend**: `npm run dev`
3. **Select Industry**: Choose your use case
4. **Configure Settings**: Adjust confidence threshold
5. **Start Detection**: Upload image or use live camera
6. **Monitor Analytics**: View business intelligence

---

**Enterprise AI Vision System** - Professional computer vision for modern businesses.
