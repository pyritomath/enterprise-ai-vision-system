import { useState, useRef, useEffect } from 'react';
import Head from 'next/head';
import { Camera, Upload, Pause, RotateCcw, Download, Settings, TrendingUp, Target, Activity, Eye, Sun, Moon, AlertTriangle, DollarSign, BarChart } from 'lucide-react';

export default function ObjectDetectionApp() {
  const [isRunning, setIsRunning] = useState(false);
  const [detections, setDetections] = useState([]);
  const [currentImage, setCurrentImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [apiStatus, setApiStatus] = useState('checking');
  const [businessAlerts, setBusinessAlerts] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);
  const [stats, setStats] = useState({
    totalDetections: 0,
    avgConfidence: 0,
    processingTime: 0,
    fps: 0,
    sessionTime: 0,
    systemStability: 0,
    detectionRate: 0
  });
  const [settings, setSettings] = useState({
    confidence: 0.5,
    industry: 'general',
    enableAnalytics: true,
    enhancedVisualization: true
  });
  const [performanceHistory, setPerformanceHistory] = useState([]);
  const [businessMetrics, setBusinessMetrics] = useState(null);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const fileInputRef = useRef(null);
  const sessionStartTime = useRef(null);

  const industryConfigs = {
    general: { 
      name: 'General Purpose', 
      objects: ['person', 'car', 'truck', 'bus', 'bottle', 'cup', 'laptop', 'cell phone'], 
      primaryColor: '#3b82f6',
      gradient: 'from-blue-500 to-indigo-600',
      bgGradient: 'from-blue-50 to-indigo-50'
    },
    retail: { 
      name: 'Retail Analytics', 
      objects: ['person', 'handbag', 'backpack', 'cell phone', 'bottle', 'cup'], 
      primaryColor: '#10b981',
      gradient: 'from-emerald-500 to-teal-600',
      bgGradient: 'from-emerald-50 to-teal-50'
    },
    security: { 
      name: 'Security & Surveillance', 
      objects: ['person', 'car', 'truck', 'backpack', 'laptop', 'knife'], 
      primaryColor: '#ef4444',
      gradient: 'from-red-500 to-rose-600',
      bgGradient: 'from-red-50 to-rose-50'
    },
    manufacturing: { 
      name: 'Manufacturing QC', 
      objects: ['bottle', 'cup', 'scissors', 'knife', 'laptop', 'book'], 
      primaryColor: '#f59e0b',
      gradient: 'from-amber-500 to-orange-600',
      bgGradient: 'from-amber-50 to-orange-50'
    }
  };

  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || (typeof window !== 'undefined' ? window.location.origin : 'http://localhost:3000');

  const currentIndustry = industryConfigs[settings.industry];

  useEffect(() => {
    if (isRunning && !sessionStartTime.current) {
      sessionStartTime.current = Date.now();
    }
    
    const interval = setInterval(() => {
      if (sessionStartTime.current) {
        const elapsed = (Date.now() - sessionStartTime.current) / 1000;
        setStats(prev => ({ ...prev, sessionTime: elapsed }));
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [isRunning]);

  // Check API status on component mount
  useEffect(() => {
    checkApiStatus();
    loadModelInfo();
  }, []);

  const checkApiStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/health`);
      if (response.ok) {
        setApiStatus('connected');
      } else {
        setApiStatus('error');
      }
    } catch (error) {
      setApiStatus('error');
    }
  };

  const loadModelInfo = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/model/info`);
      if (response.ok) {
        const info = await response.json();
        setModelInfo(info);
      }
    } catch (error) {
      console.error('Failed to load model info:', error);
    }
  };

  const startCamera = async () => {
    try {
      setIsProcessing(true);
      
      // Check if mediaDevices is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera API not supported in this browser');
      }
      
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: { ideal: 1280 }, 
          height: { ideal: 720 },
          frameRate: { ideal: 30 }
        }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsRunning(true);
        sessionStartTime.current = Date.now();
        processVideo();
      }
    } catch (error) {
      console.error('Camera access error:', error);
      
      let errorMessage = 'Camera access failed.';
      
      if (error.name === 'NotAllowedError') {
        errorMessage = 'Camera access denied. Please allow camera permissions and try again.';
      } else if (error.name === 'NotFoundError') {
        errorMessage = 'No camera found. Please connect a camera and try again.';
      } else if (error.name === 'NotReadableError') {
        errorMessage = 'Camera is being used by another application. Please close other apps and try again.';
      } else if (error.name === 'OverconstrainedError') {
        errorMessage = 'Camera constraints not supported. Trying with basic settings...';
        
        // Try with basic constraints
        try {
          const basicStream = await navigator.mediaDevices.getUserMedia({ video: true });
          if (videoRef.current) {
            videoRef.current.srcObject = basicStream;
            streamRef.current = basicStream;
            setIsRunning(true);
            sessionStartTime.current = Date.now();
            processVideo();
            return;
          }
        } catch (basicError) {
          errorMessage = 'Camera access failed with basic settings.';
        }
      }
      
      alert(errorMessage);
    } finally {
      setIsProcessing(false);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsRunning(false);
  };

  const processVideo = () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    const processFrame = async () => {
      if (!isRunning || !video.videoWidth || !video.videoHeight) {
        if (isRunning) {
          setTimeout(() => requestAnimationFrame(processFrame), 100);
        }
        return;
      }
      
      const startTime = performance.now();
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      const detections = await performObjectDetection(canvas);
      drawDetections(ctx, detections);
      
      const processingTime = performance.now() - startTime;
      updateStats(detections, processingTime);
      
      setTimeout(() => requestAnimationFrame(processFrame), 100);
    };
    
    processFrame();
  };

  const performObjectDetection = async (canvas) => {
    try {
      // Convert canvas to base64
      const imageData = canvas.toDataURL('image/jpeg', 0.8);
      
      const response = await fetch(`${API_BASE_URL}/api/detect/base64`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageData,
          confidence: settings.confidence,
          industry: settings.industry
        })
      });
      
      if (!response.ok) {
        throw new Error('API request failed');
      }
      
      const result = await response.json();
      
      // Update performance metrics
      if (result.performance) {
        setPerformanceHistory(prev => [...prev.slice(-99), result.performance]);
        setStats(prev => ({
          ...prev,
          processingTime: result.performance.total_time * 1000,
          fps: result.performance.processing_fps
        }));
      }
      
      // Update business alerts
      if (result.business_alerts) {
        setBusinessAlerts(prev => [...prev.slice(-19), ...result.business_alerts]);
      }
      
      return result.detections || [];
      
    } catch (error) {
      console.error('Detection API error:', error);
      return [];
    }
  };

  const drawDetections = (ctx, detections) => {
    ctx.font = '14px Arial';
    ctx.textAlign = 'start';
    
    detections.forEach(detection => {
      const { bbox, class: className, confidence, priority } = detection;
      const color = priority === 'high' ? '#10b981' : priority === 'medium' ? '#f59e0b' : '#ef4444';
      
      // Draw bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);
      
      // Draw label background
      const label = `${className}: ${(confidence * 100).toFixed(1)}%`;
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = color;
      ctx.fillRect(bbox.x, bbox.y - 25, textWidth + 10, 20);
      
      // Draw label text
      ctx.fillStyle = 'white';
      ctx.fillText(label, bbox.x + 5, bbox.y - 10);
      
      // Draw corner indicators
      const cornerSize = 15;
      ctx.lineWidth = 4;
      // Top-left corner
      ctx.beginPath();
      ctx.moveTo(bbox.x, bbox.y + cornerSize);
      ctx.lineTo(bbox.x, bbox.y);
      ctx.lineTo(bbox.x + cornerSize, bbox.y);
      ctx.stroke();
      
      // Top-right corner
      ctx.beginPath();
      ctx.moveTo(bbox.x + bbox.width - cornerSize, bbox.y);
      ctx.lineTo(bbox.x + bbox.width, bbox.y);
      ctx.lineTo(bbox.x + bbox.width, bbox.y + cornerSize);
      ctx.stroke();
      
      // Bottom-left corner
      ctx.beginPath();
      ctx.moveTo(bbox.x, bbox.y + bbox.height - cornerSize);
      ctx.lineTo(bbox.x, bbox.y + bbox.height);
      ctx.lineTo(bbox.x + cornerSize, bbox.y + bbox.height);
      ctx.stroke();
      
      // Bottom-right corner
      ctx.beginPath();
      ctx.moveTo(bbox.x + bbox.width - cornerSize, bbox.y + bbox.height);
      ctx.lineTo(bbox.x + bbox.width, bbox.y + bbox.height);
      ctx.lineTo(bbox.x + bbox.width, bbox.y + bbox.height - cornerSize);
      ctx.stroke();
    });
    
    setDetections(detections);
  };

  const updateStats = (newDetections, processingTime) => {
    setStats(prev => ({
      ...prev,
      totalDetections: prev.totalDetections + newDetections.length,
      avgConfidence: newDetections.length > 0 ? 
        newDetections.reduce((sum, d) => sum + d.confidence, 0) / newDetections.length : 
        prev.avgConfidence,
      processingTime: processingTime,
      fps: Math.round(1000 / processingTime)
    }));
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      setIsProcessing(true);
      
      try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('confidence', settings.confidence);
        formData.append('industry', settings.industry);
        
        const response = await fetch(`${API_BASE_URL}/api/detect`, {
          method: 'POST',
          body: formData
        });
        
        if (!response.ok) {
          throw new Error('Upload failed');
        }
        
        const result = await response.json();
        
        // Display the annotated image
        if (result.annotated_image) {
          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d');
          const img = new Image();
          img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
          };
          img.src = `data:image/jpeg;base64,${result.annotated_image}`;
        }
        
        // Update detections and stats
        if (result.detections) {
          setDetections(result.detections);
          updateStats(result.detections, result.performance?.total_time || 0);
        }
        
        // Update business alerts
        if (result.business_alerts) {
          setBusinessAlerts(prev => [...prev.slice(-19), ...result.business_alerts]);
        }
        
        // Update performance history
        if (result.performance) {
          setPerformanceHistory(prev => [...prev.slice(-99), result.performance]);
        }
        
      } catch (error) {
        console.error('File upload error:', error);
        alert('Failed to process image. Please try again.');
      } finally {
        setIsProcessing(false);
      }
    }
  };

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const calculateBusinessMetrics = () => {
    if (!detections.length) return null;
    
    const uniqueObjects = new Set(detections.map(d => d.class)).size;
    const totalValue = detections.reduce((sum, d) => {
      return sum + (typeof d.revenue_value === 'number' ? d.revenue_value : 0);
    }, 0);
    
    const avgConfidence = detections.reduce((sum, d) => sum + d.confidence, 0) / detections.length;
    const highConfidenceCount = detections.filter(d => d.confidence > 0.8).length;
    
    return {
      totalValue,
      uniqueObjects,
      avgConfidence,
      highConfidenceRatio: highConfidenceCount / detections.length,
      detectionRate: detections.length / Math.max(stats.sessionTime, 1) * 60 // per minute
    };
  };

  return (
    <div className={`min-h-screen transition-all duration-300 ${
      darkMode 
        ? 'bg-gradient-to-br from-gray-900 to-gray-800' 
        : `bg-gradient-to-br ${currentIndustry.bgGradient}`
    }`}>
      <Head>
        <title>AI Object Detection - Professional Computer Vision</title>
        <meta name="description" content="Advanced AI object detection system" />
      </Head>

      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-12">
          <div className="flex justify-center items-center gap-4 mb-6">
            <h1 className={`text-6xl font-bold bg-gradient-to-r ${
              darkMode 
                ? 'from-gray-100 to-gray-300' 
                : 'from-gray-900 to-gray-700'
            } bg-clip-text text-transparent`}>
              Enterprise AI Vision
            </h1>
            <button
              onClick={() => setDarkMode(!darkMode)}
              className={`p-3 rounded-full transition-all duration-300 ${
                darkMode 
                  ? 'bg-gray-700 hover:bg-gray-600 text-yellow-400' 
                  : 'bg-gray-200 hover:bg-gray-300 text-gray-700'
              }`}
            >
              {darkMode ? <Sun className="w-6 h-6" /> : <Moon className="w-6 h-6" />}
            </button>
          </div>
          <p className={`text-xl mb-4 ${
            darkMode ? 'text-gray-300' : 'text-gray-600'
          }`}>
            Professional YOLOv8 computer vision with business intelligence
          </p>
          
          {/* API Status Indicator */}
          <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium ${
            apiStatus === 'connected' 
              ? 'bg-green-100 text-green-800' 
              : apiStatus === 'error' 
              ? 'bg-red-100 text-red-800' 
              : 'bg-yellow-100 text-yellow-800'
          }`}>
            <div className={`w-2 h-2 rounded-full ${
              apiStatus === 'connected' 
                ? 'bg-green-500' 
                : apiStatus === 'error' 
                ? 'bg-red-500' 
                : 'bg-yellow-500'
            }`}></div>
            API Status: {apiStatus === 'connected' ? 'Connected' : apiStatus === 'error' ? 'Disconnected' : 'Checking...'}
          </div>
          
          <div className="flex justify-center gap-4 mb-8">
            <div className={`px-4 py-2 rounded-full text-sm font-medium ${
              isRunning ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'
            }`}>
              {isRunning ? 'Live Processing' : 'Ready'}
            </div>
            <div className={`px-4 py-2 rounded-full text-sm font-medium bg-gradient-to-r ${currentIndustry.gradient} text-white`}>
              {currentIndustry.name}
            </div>
            {modelInfo && (
              <div className="px-4 py-2 rounded-full text-sm font-medium bg-blue-100 text-blue-800">
                YOLOv8 • {modelInfo.device.toUpperCase()} • {modelInfo.theoretical_max_fps} FPS
              </div>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-4 gap-8">
          <div className="xl:col-span-3">
            <div className={`backdrop-blur-sm rounded-2xl shadow-2xl p-6 ${
              darkMode 
                ? 'bg-gray-800/90 border border-gray-700' 
                : 'bg-white/80'
            }`}>
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <Eye className="w-6 h-6 text-blue-600" />
                  <div>
                    <h2 className={`text-2xl font-bold ${
                      darkMode ? 'text-gray-100' : 'text-gray-900'
                    }`}>Live Detection</h2>
                    <p className={`${
                      darkMode ? 'text-gray-300' : 'text-gray-600'
                    }`}>Real-time object recognition</p>
                  </div>
                </div>
                
                <div className="flex gap-2">
                  <button
                    onClick={startCamera}
                    disabled={isRunning || isProcessing || apiStatus !== 'connected'}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-300 ${
                      isRunning || isProcessing || apiStatus !== 'connected' 
                        ? `${darkMode ? 'bg-gray-700 text-gray-500' : 'bg-gray-100 text-gray-400'}` 
                        : 'bg-blue-600 text-white hover:bg-blue-700 transform hover:scale-105'
                    }`}
                  >
                    <Camera className={`w-4 h-4 ${isProcessing ? 'animate-pulse' : ''}`} />
                    {isRunning ? 'Running' : isProcessing ? 'Starting...' : apiStatus !== 'connected' ? 'API Offline' : 'Start Camera'}
                  </button>
                  
                  <button
                    onClick={stopCamera}
                    disabled={!isRunning}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-300 ${
                      !isRunning 
                        ? `${darkMode ? 'bg-gray-700 text-gray-500' : 'bg-gray-100 text-gray-400'}` 
                        : 'bg-red-600 text-white hover:bg-red-700 transform hover:scale-105'
                    }`}
                  >
                    <Pause className="w-4 h-4" />
                    Stop
                  </button>
                  
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-all duration-300 transform hover:scale-105"
                  >
                    <Upload className="w-4 h-4" />
                    Upload
                  </button>
                </div>
              </div>

              <div className={`relative rounded-xl overflow-hidden ${
                darkMode ? 'bg-gray-900 border border-gray-700' : 'bg-gray-900'
              }`}>
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="absolute inset-0 w-full h-full object-cover"
                  style={{ display: isRunning ? 'block' : 'none' }}
                />
                <canvas
                  ref={canvasRef}
                  className="w-full h-auto max-h-[600px] object-contain"
                />
                
                {!isRunning && !currentImage && (
                  <div className="absolute inset-0 flex items-center justify-center text-white">
                    <div className="text-center">
                      <Camera className="w-16 h-16 mx-auto mb-4 opacity-70 animate-pulse" />
                      <p className="text-xl font-semibold">Ready for Detection</p>
                      <p className="text-gray-300">Start camera or upload an image</p>
                    </div>
                  </div>
                )}
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileUpload}
                className="hidden"
              />
            </div>
          </div>

          <div className="xl:col-span-1 space-y-6">
            <div className={`backdrop-blur-sm rounded-2xl shadow-2xl p-6 ${
              darkMode 
                ? 'bg-gray-800/90 border border-gray-700' 
                : 'bg-white/80'
            }`}>
              <div className="flex items-center gap-3 mb-6">
                <Settings className="w-5 h-5 text-purple-600" />
                <h3 className={`text-lg font-bold ${
                  darkMode ? 'text-gray-100' : 'text-gray-900'
                }`}>Configuration</h3>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className={`block text-sm font-medium mb-2 ${
                    darkMode ? 'text-gray-300' : 'text-gray-700'
                  }`}>Industry Focus</label>
                  <select
                    value={settings.industry}
                    onChange={(e) => setSettings(prev => ({ ...prev, industry: e.target.value }))}
                    className={`w-full border rounded-lg px-3 py-2 transition-all duration-300 ${
                      darkMode 
                        ? 'bg-gray-700 border-gray-600 text-gray-100 focus:border-blue-500' 
                        : 'bg-white border-gray-300 text-gray-900 focus:border-blue-500'
                    }`}
                  >
                    {Object.entries(industryConfigs).map(([key, config]) => (
                      <option key={key} value={key}>{config.name}</option>
                    ))}
                  </select>
                  
                  <div className={`mt-2 text-xs ${
                    darkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>
                    Objects: {currentIndustry.objects.slice(0, 3).join(', ')}...
                  </div>
                </div>

                <div>
                  <label className={`block text-sm font-medium mb-2 ${
                    darkMode ? 'text-gray-300' : 'text-gray-700'
                  }`}>
                    Confidence: {(settings.confidence * 100).toFixed(0)}%
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="0.95"
                    step="0.05"
                    value={settings.confidence}
                    onChange={(e) => setSettings(prev => ({ ...prev, confidence: parseFloat(e.target.value) }))}
                    className={`w-full slider accent-blue-600 ${
                      darkMode ? 'bg-gray-700' : 'bg-gray-200'
                    }`}
                  />
                </div>
              </div>
            </div>

            <div className={`backdrop-blur-sm rounded-2xl shadow-2xl p-6 ${
              darkMode 
                ? 'bg-gray-800/90 border border-gray-700' 
                : 'bg-white/80'
            }`}>
              <div className="flex items-center gap-3 mb-6">
                <TrendingUp className="w-5 h-5 text-green-600" />
                <h3 className={`text-lg font-bold ${
                  darkMode ? 'text-gray-100' : 'text-gray-900'
                }`}>Performance</h3>
              </div>
              
              <div className="grid grid-cols-2 gap-3">
                <div className={`text-center p-4 rounded-xl transition-all duration-300 ${
                  darkMode ? 'bg-blue-900/50 border border-blue-700' : 'bg-blue-50'
                }`}>
                  <div className="text-2xl font-bold text-blue-600">{stats.totalDetections}</div>
                  <div className={`text-xs ${
                    darkMode ? 'text-gray-300' : 'text-gray-600'
                  }`}>Objects</div>
                </div>
                <div className={`text-center p-4 rounded-xl transition-all duration-300 ${
                  darkMode ? 'bg-green-900/50 border border-green-700' : 'bg-green-50'
                }`}>
                  <div className="text-2xl font-bold text-green-600">{(stats.avgConfidence * 100).toFixed(1)}%</div>
                  <div className={`text-xs ${
                    darkMode ? 'text-gray-300' : 'text-gray-600'
                  }`}>Confidence</div>
                </div>
                <div className={`text-center p-4 rounded-xl transition-all duration-300 ${
                  darkMode ? 'bg-purple-900/50 border border-purple-700' : 'bg-purple-50'
                }`}>
                  <div className="text-2xl font-bold text-purple-600">{stats.processingTime.toFixed(0)}ms</div>
                  <div className={`text-xs ${
                    darkMode ? 'text-gray-300' : 'text-gray-600'
                  }`}>Processing</div>
                </div>
                <div className={`text-center p-4 rounded-xl transition-all duration-300 ${
                  darkMode ? 'bg-orange-900/50 border border-orange-700' : 'bg-orange-50'
                }`}>
                  <div className="text-2xl font-bold text-orange-600">{stats.fps}</div>
                  <div className={`text-xs ${
                    darkMode ? 'text-gray-300' : 'text-gray-600'
                  }`}>FPS</div>
                </div>
              </div>

              <div className={`mt-4 p-3 rounded-lg ${
                darkMode ? 'bg-gray-700/50 border border-gray-600' : 'bg-gray-50'
              }`}>
                <div className="flex justify-between">
                  <span className={`text-sm font-medium ${
                    darkMode ? 'text-gray-300' : 'text-gray-700'
                  }`}>Session Time</span>
                  <span className={`text-sm font-bold ${
                    darkMode ? 'text-gray-100' : 'text-gray-900'
                  }`}>{formatTime(stats.sessionTime)}</span>
                </div>
              </div>
            </div>

            {detections.length > 0 && (
              <div className={`backdrop-blur-sm rounded-2xl shadow-2xl p-6 ${
                darkMode 
                  ? 'bg-gray-800/90 border border-gray-700' 
                  : 'bg-white/80'
              }`}>
                <div className="flex items-center gap-3 mb-4">
                  <Target className="w-5 h-5 text-indigo-600" />
                  <h3 className={`text-lg font-bold ${
                    darkMode ? 'text-gray-100' : 'text-gray-900'
                  }`}>Recent Detections</h3>
                </div>
                
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {detections.slice(-10).map((detection, index) => (
                    <div key={detection.id || index} className={`flex items-center justify-between p-3 rounded-lg transition-all duration-300 ${
                      darkMode ? 'bg-gray-700/50 border border-gray-600' : 'bg-gray-50'
                    }`}>
                      <div className="flex items-center gap-3">
                        <span className={`text-sm font-medium ${
                          darkMode ? 'text-gray-200' : 'text-gray-900'
                        }`}>{detection.class}</span>
                        <span className={`text-xs px-2 py-1 rounded-full ${
                          detection.priority === 'high' ? 'bg-green-100 text-green-800' :
                          detection.priority === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}>{detection.priority}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className={`text-xs ${
                          darkMode ? 'text-gray-400' : 'text-gray-600'
                        }`}>
                          {typeof detection.revenue_value === 'number' ? `$${detection.revenue_value}` : detection.revenue_value}
                        </span>
                        <span className={`text-sm px-2 py-1 rounded-full ${
                          detection.confidence > 0.8 ? 'bg-green-100 text-green-800' :
                          detection.confidence > 0.6 ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}>{(detection.confidence * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {/* Business Intelligence Panel */}
            {calculateBusinessMetrics() && (
              <div className={`backdrop-blur-sm rounded-2xl shadow-2xl p-6 ${
                darkMode 
                  ? 'bg-gray-800/90 border border-gray-700' 
                  : 'bg-white/80'
              }`}>
                <div className="flex items-center gap-3 mb-4">
                  <DollarSign className="w-5 h-5 text-green-600" />
                  <h3 className={`text-lg font-bold ${
                    darkMode ? 'text-gray-100' : 'text-gray-900'
                  }`}>Business Intelligence</h3>
                </div>
                
                <div className="space-y-3">
                  <div className={`p-3 rounded-lg ${
                    darkMode ? 'bg-gray-700/50' : 'bg-green-50'
                  }`}>
                    <div className="text-2xl font-bold text-green-600">
                      ${calculateBusinessMetrics().totalValue.toFixed(0)}
                    </div>
                    <div className={`text-sm ${
                      darkMode ? 'text-gray-300' : 'text-gray-600'
                    }`}>Session Value</div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-2">
                    <div className={`p-2 rounded-lg text-center ${
                      darkMode ? 'bg-gray-700/50' : 'bg-blue-50'
                    }`}>
                      <div className="text-lg font-bold text-blue-600">
                        {calculateBusinessMetrics().uniqueObjects}
                      </div>
                      <div className={`text-xs ${
                        darkMode ? 'text-gray-400' : 'text-gray-600'
                      }`}>Object Types</div>
                    </div>
                    <div className={`p-2 rounded-lg text-center ${
                      darkMode ? 'bg-gray-700/50' : 'bg-purple-50'
                    }`}>
                      <div className="text-lg font-bold text-purple-600">
                        {calculateBusinessMetrics().detectionRate.toFixed(1)}
                      </div>
                      <div className={`text-xs ${
                        darkMode ? 'text-gray-400' : 'text-gray-600'
                      }`}>Objects/min</div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {/* Business Alerts */}
            {businessAlerts.length > 0 && (
              <div className={`backdrop-blur-sm rounded-2xl shadow-2xl p-6 ${
                darkMode 
                  ? 'bg-gray-800/90 border border-gray-700' 
                  : 'bg-white/80'
              }`}>
                <div className="flex items-center gap-3 mb-4">
                  <AlertTriangle className="w-5 h-5 text-orange-600" />
                  <h3 className={`text-lg font-bold ${
                    darkMode ? 'text-gray-100' : 'text-gray-900'
                  }`}>Business Alerts</h3>
                </div>
                
                <div className="space-y-2 max-h-32 overflow-y-auto">
                  {businessAlerts.slice(-5).map((alert, index) => (
                    <div key={index} className={`flex items-center justify-between p-2 rounded-lg ${
                      darkMode ? 'bg-orange-900/20 border border-orange-700' : 'bg-orange-50 border border-orange-200'
                    }`}>
                      <span className={`text-sm font-medium ${
                        darkMode ? 'text-orange-200' : 'text-orange-800'
                      }`}>
                        {alert.class} detected
                      </span>
                      <span className={`text-xs px-2 py-1 rounded-full ${
                        darkMode ? 'bg-orange-800 text-orange-200' : 'bg-orange-100 text-orange-800'
                      }`}>
                        {(alert.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        <div className={`mt-12 text-center py-12 rounded-2xl transition-all duration-300 ${
          darkMode 
            ? 'bg-gray-800/90 border border-gray-700 text-gray-100' 
            : 'bg-gray-900 text-white'
        }`}>
          <h3 className="text-2xl font-bold mb-4">Enterprise AI Vision System</h3>
          <p className={`mb-6 ${
            darkMode ? 'text-gray-300' : 'text-gray-300'
          }`}>
            YOLOv8 computer vision with FastAPI backend and business intelligence
          </p>
          <div className="flex justify-center gap-4 text-sm flex-wrap">
            <span className="px-3 py-1 bg-blue-600 rounded-full">Next.js Frontend</span>
            <span className="px-3 py-1 bg-green-600 rounded-full">FastAPI Backend</span>
            <span className="px-3 py-1 bg-purple-600 rounded-full">YOLOv8 Detection</span>
            <span className="px-3 py-1 bg-orange-600 rounded-full">Business Analytics</span>
            <span className="px-3 py-1 bg-red-600 rounded-full">Real-time Processing</span>
          </div>
          
          {modelInfo && (
            <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 bg-gray-800 rounded-lg">
                <div className="text-lg font-bold text-blue-400">{modelInfo.total_parameters?.toLocaleString()}</div>
                <div className="text-sm text-gray-300">Parameters</div>
              </div>
              <div className="p-4 bg-gray-800 rounded-lg">
                <div className="text-lg font-bold text-green-400">{modelInfo.model_size_mb} MB</div>
                <div className="text-sm text-gray-300">Model Size</div>
              </div>
              <div className="p-4 bg-gray-800 rounded-lg">
                <div className="text-lg font-bold text-purple-400">{modelInfo.num_classes}</div>
                <div className="text-sm text-gray-300">Object Classes</div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
