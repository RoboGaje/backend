# 🔧 Backend - Realtime Face Detection & Crowd Analysis API

Backend FastAPI untuk realtime face detection dan crowd analysis dengan WebSocket support.

## 🎯 Features

### 🚀 **WebSocket API**
- Real-time frame processing
- Low-latency detection results
- Connection management
- Error handling dan reconnection

### 🔍 **Detection Services**
- YOLO face detection
- Crowd density analysis
- Bounding box coordinates
- Confidence scores

### 📊 **Analytics**
- Real-time statistics
- Crowd level classification
- Performance metrics
- Historical data logging

### ⚠️ **Alert System**
- Crowd threshold monitoring
- Real-time notifications
- Configurable alert levels

## 🛠️ Technology Stack

- **Framework**: FastAPI
- **WebSocket**: Socket.IO (python-socketio)
- **ML Framework**: PyTorch + Ultralytics YOLO
- **Image Processing**: OpenCV, PIL
- **Async**: asyncio, uvicorn
- **Database**: SQLite (development) / PostgreSQL (production)
- **Caching**: Redis (optional)
- **Monitoring**: Prometheus + Grafana (optional)

## 📁 Project Structure

```
backend/
├── app/
│   ├── main.py                # FastAPI application entry point
│   ├── config.py              # Configuration settings
│   ├── api/
│   │   ├── __init__.py
│   │   ├── websocket.py       # WebSocket handlers
│   │   ├── detection.py       # Detection endpoints
│   │   └── health.py          # Health check endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── detection_service.py # YOLO detection logic
│   │   ├── crowd_analysis.py   # Crowd analysis algorithms
│   │   ├── frame_processor.py  # Frame processing utilities
│   │   └── alert_service.py    # Alert management
│   ├── models/
│   │   ├── __init__.py
│   │   ├── detection.py        # Pydantic models
│   │   └── response.py         # Response models
│   ├── core/
│   │   ├── __init__.py
│   │   ├── model_loader.py     # YOLO model loading
│   │   ├── websocket_manager.py # WebSocket connection manager
│   │   └── exceptions.py       # Custom exceptions
│   └── utils/
│       ├── __init__.py
│       ├── image_utils.py      # Image processing utilities
│       ├── logging.py          # Logging configuration
│       └── metrics.py          # Performance metrics
├── tests/                     # Unit tests
├── requirements.txt           # Dependencies
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker compose for development
└── README.md                  # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA (optional, untuk GPU acceleration)
- Models dari `../train-eval-model/`

### Installation
```bash
cd backend
pip install -r requirements.txt
```

### Model Setup
```bash
# Copy models dari train-eval-model
cp -r ../train-eval-model/models ./models
cp -r ../train-eval-model/YOLO12n_pretrained ./models/
```

### Development
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 🔌 WebSocket API

### Connection
```
ws://localhost:8000/ws
```

### Events

#### Client → Server
```json
{
  "event": "process_frame",
  "data": {
    "frame": "base64_encoded_image",
    "timestamp": 1640995200000,
    "settings": {
      "confidence_threshold": 0.5,
      "crowd_threshold": 10
    }
  }
}
```

#### Server → Client
```json
{
  "event": "detection_result",
  "data": {
    "faces": [
      {
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.95
      }
    ],
    "bodies": [
      {
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.87
      }
    ],
    "crowd_analysis": {
      "face_count": 3,
      "body_count": 4,
      "crowd_level": "medium",
      "density_score": 0.6
    },
    "alerts": [
      {
        "type": "crowd_warning",
        "message": "Crowd threshold exceeded",
        "level": "warning"
      }
    ],
    "processing_time": 0.045,
    "timestamp": 1640995200100
  }
}
```

#### Connection Events
```json
// Connection established
{
  "event": "connected",
  "data": {
    "client_id": "uuid",
    "server_time": 1640995200000
  }
}

// Error handling
{
  "event": "error",
  "data": {
    "code": "PROCESSING_ERROR",
    "message": "Failed to process frame",
    "details": "Model inference failed"
  }
}
```

## 🔍 Detection Pipeline

### Frame Processing Flow
```
Base64 Image → Decode → Resize → YOLO Inference → Post-processing → Results
```

### YOLO Models Integration
- **Face Detection**: Model dari `../train-eval-model/YOLO12n_finetuned/`
- **Body Detection**: Model dari `../train-eval-model/YOLO12n_pretrained/`
- **TensorRT**: Support untuk optimized inference

### Crowd Analysis Algorithm
```python
def analyze_crowd(face_count, body_count):
    total_people = max(face_count, body_count)
    
    if total_people == 0:
        return "empty"
    elif total_people <= 2:
        return "low"
    elif total_people <= 5:
        return "medium"
    elif total_people <= 10:
        return "high"
    else:
        return "very_high"
```

## ⚙️ Configuration

### Environment Variables
```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=False

# Model Paths
FACE_MODEL_PATH=./models/YOLO12n_finetuned/best.pt
BODY_MODEL_PATH=./models/YOLO12n_pretrained/yolo12n.pt

# Detection Settings
CONFIDENCE_THRESHOLD=0.5
CROWD_THRESHOLD=10
MAX_FRAME_SIZE=1024

# Performance
USE_GPU=True
TENSORRT_ENABLED=False
MAX_CONCURRENT_CONNECTIONS=50

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

### Model Configuration
```python
# app/config.py
class Settings:
    # Model settings
    face_model_path: str = "./models/YOLO12n_finetuned/best.pt"
    body_model_path: str = "./models/YOLO12n_pretrained/yolo12n.pt"
    
    # Detection settings
    confidence_threshold: float = 0.5
    crowd_threshold: int = 10
    
    # Performance settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_frame_size: int = 1024
```

## 📊 API Endpoints

### Health Check
```
GET /health
Response: {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}
```

### Model Info
```
GET /api/models/info
Response: {
  "face_model": {"loaded": true, "path": "...", "device": "cuda"},
  "body_model": {"loaded": true, "path": "...", "device": "cuda"}
}
```

### Statistics
```
GET /api/stats
Response: {
  "active_connections": 5,
  "total_frames_processed": 12450,
  "average_processing_time": 0.045,
  "uptime": 3600
}
```

## 🚨 Alert System

### Alert Types
- **crowd_warning**: Crowd threshold exceeded
- **processing_error**: Model inference failed
- **connection_limit**: Too many concurrent connections

### Alert Levels
- **info**: Informational messages
- **warning**: Warning conditions
- **error**: Error conditions
- **critical**: Critical system issues

## 🔧 Performance Optimization

### Model Optimization
- Model loading pada startup
- GPU memory management
- Batch processing untuk multiple frames
- TensorRT optimization (optional)

### WebSocket Optimization
- Connection pooling
- Frame rate limiting
- Compression untuk large payloads
- Graceful disconnection handling

### Memory Management
- Frame buffer management
- Model cache optimization
- Garbage collection tuning

## 🐳 Docker Support

### Dockerfile
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  backend:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - DEBUG=True
      - USE_GPU=False
```

## 📈 Monitoring

### Metrics
- Request latency
- Frame processing time
- Memory usage
- GPU utilization
- Active connections

### Logging
- Structured logging dengan JSON format
- Request/response logging
- Error tracking
- Performance metrics

## 🔗 Integration

Backend ini terintegrasi dengan:
- **Frontend**: Next.js application via WebSocket
- **ML Models**: YOLO models dari `../train-eval-model/`
- **Monitoring**: Prometheus, Grafana (optional)
- **Database**: SQLite/PostgreSQL untuk logging dan analytics

## 🚀 Deployment

### Production Checklist
- [ ] Environment variables configured
- [ ] Models copied dan tested
- [ ] SSL/TLS certificates (untuk HTTPS/WSS)
- [ ] Reverse proxy setup (Nginx)
- [ ] Monitoring dan logging configured
- [ ] Database setup (jika menggunakan PostgreSQL)
- [ ] Performance testing completed 