# 🌿 Plant Diagnostic System (MiniGPT‑v2 + ResNet)

An advanced **plant diagnostic system** that combines a **ResNet-50** classifier for accurate disease identification with **MiniGPT‑v2** for detailed medical reporting. The system provides comprehensive strawberry plant health analysis with doctor-grade diagnostic reports, confidence scoring, and actionable treatment recommendations.

> Research prototype. Not medical/agronomic advice.

---

## ✨ Features

- **Dual AI Architecture**: ResNet-50 classifier + MiniGPT‑v2 vision-language model
- **7-Class Disease Detection**: healthy, overwatering, root rot, drought, frost injury, gray mold, white mold
- **Ground Truth Approach**: ResNet diagnosis is treated as absolute truth, MiniGPT explains the evidence
- **Confidence Scoring**: Visual confidence indicators (🟢 ≥90%, 🟡 70-90%, 🔴 <70%)
- **Doctor-Grade Reports**: Structured medical reports with diagnosis, visible cues, and recommendations
- **Interactive Knowledge Graph**: FAOSTAT agricultural data visualization
- **Modern Web Interface**: Dark theme with responsive Gradio UI
- **Real-time Processing**: Optimized for fast inference on single GPU
- **📷 Webcam Integration**: Live camera feed for instant plant diagnosis
- **🔍 Enhanced Analysis**: Web research integration for comprehensive treatment recommendations

---

## 🧰 Requirements

- Python **3.8+**
- **CUDA GPU** (RTX 3090/4090 recommended for optimal performance)
- **PyTorch** with CUDA support
- Model weights (automatically loaded from configured paths):
  - LLaMA‑2‑7B chat weights: `llama_weights/Llama-2-7b-chat-hf/`
  - MiniGPT‑v2 checkpoint: `output/minigptv2_strawberry_diagnostic/*/checkpoint_best.pth`
  - ResNet-50 classifier: `plant_diagnostic/models/resnet_straw_final.pth`
- **8GB+ VRAM** for smooth operation
- (Optional) **SERPAPI** key for enhanced web search features

---

## ⚙️ Setup

```bash
# Clone the repository
git clone https://github.com/your-username/plant-diagnostic-system.git
cd plant-diagnostic-system

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install required dependencies
pip install transformers==4.41.1 bitsandbytes gradio pillow numpy pandas plotly networkx python-dotenv timm

# Optional: Install additional dependencies for enhanced features
pip install serpapi  # For web search features
```

**Environment Setup:**
```bash
# Set CUDA memory allocation for better performance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional: Set SERPAPI key for enhanced features
export SERP_API_KEY=your_serpapi_key_here
```

---

## 🔧 Configuration

The system uses YAML configuration files for easy setup. Key configuration files:

- **`eval_configs/minigptv2_eval.yaml`** - Demo/inference configuration
- **`train_configs/minigptv2_strawberry_diagnostic.yaml`** - Training configuration

**Model Paths** (automatically configured):
- MiniGPT-v2 checkpoint: `output/minigptv2_strawberry_diagnostic/*/checkpoint_best.pth`
- LLaMA-2-7B weights: `llama_weights/Llama-2-7b-chat-hf/`
- ResNet-50 classifier: `plant_diagnostic/models/resnet_straw_final.pth`

**Key Configuration Parameters:**
```yaml
model:
  arch: minigpt_v2
  ckpt: /path/to/checkpoint_best.pth
  lora_r: 16
  lora_alpha: 32

run:
  evaluate: true
  val_splits: ["val"]
  max_val_steps: 50
  auto_val_split_ratio: 0.1
```

---

## 🚀 Running the System

### **Quick Start with Launch Script**

The easiest way to run the system:

```bash
# Basic usage
./launch_demo_v5.sh

# With public share link
./launch_demo_v5.sh --share

# With ResNet anchor for faster first inference
./launch_demo_v5.sh --share --resnet-anchor

# Use specific GPU
./launch_demo_v5.sh --gpu 1 --share
```

### **Manual Launch**

For more control, run directly:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python demo_v5.py --cfg-path eval_configs/minigptv2_eval.yaml --resnet-anchor
```

**Parameters:**
- `--cfg-path`: Points to the evaluation configuration
- `--resnet-anchor`: Preloads ResNet for faster first inference
- `--share`: Create public share link (accessible from anywhere)
- `--gpu-id`: Specify GPU device ID
- `CUDA_VISIBLE_DEVICES=0`: Uses GPU 0 (adjust as needed)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: Optimizes CUDA memory allocation

The system will start a Gradio web interface. Open the provided URL in your browser.

### **📷 Webcam Demo**

Experience real-time plant diagnosis with live camera feed:

[![Webcam Demo](https://img.youtube.com/vi/PXN6_6oj7_M/0.jpg)](https://youtu.be/PXN6_6oj7_M)

**Watch the Demo Video**: [https://youtu.be/PXN6_6oj7_M](https://youtu.be/PXN6_6oj7_M)

**Webcam Features:**
- **Live Capture**: Click the camera icon to capture images directly from your webcam
- **Instant Analysis**: Get immediate diagnosis results from captured images
- **Real-time Processing**: Optimized for fast inference on live camera feed
- **Browser Integration**: Works seamlessly with modern browsers (Chrome, Firefox, Safari)

**How to Use Webcam:**
1. Launch the system with `./launch_demo_v5.sh --share`
2. Open the provided URL in your browser
3. Click the camera icon in the image upload area
4. Allow camera access when prompted
5. Click the camera icon again to capture
6. Get instant AI-powered plant diagnosis!

---

## 🎥 Videos

### **System Demonstration Videos**

#### **Plant Diagnostic System - Full Demo**
[![Plant Diagnostic System - Full Demo](https://img.youtube.com/vi/eduSbkigjLY/0.jpg)](https://youtu.be/eduSbkigjLY?si=hqwsnJdp-dE8X3sd)

**Watch**: [https://youtu.be/eduSbkigjLY](https://youtu.be/eduSbkigjLY?si=hqwsnJdp-dE8X3sd)

Comprehensive demonstration of the Plant Diagnostic System showing:
- Complete workflow from image capture to diagnosis
- ResNet classification and MiniGPT-v2 explanation
- Knowledge graph visualization
- User interface features and capabilities

#### **Advanced Features & Webcam Integration**
[![Advanced Features & Webcam Integration](https://img.youtube.com/vi/-QEf8KkALK4/0.jpg)](https://youtu.be/-QEf8KkALK4?si=Yy40iCD6IiNp1YLl)

**Watch**: [https://youtu.be/-QEf8KkALK4](https://youtu.be/-QEf8KkALK4?si=Yy40iCD6IiNp1YLl)

Deep dive into advanced system features including:
- Webcam integration and real-time capture
- Enhanced analysis with web research
- Confidence scoring and medical report generation
- Troubleshooting and best practices

### **Training Mode**

Train the MiniGPT-v2 model on strawberry diagnostic data:

```bash
CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29607 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m torch.distributed.run --nproc_per_node=2 train.py \
  --cfg-path train_configs/minigptv2_strawberry_diagnostic.yaml
```

**Training Features:**
- **Distributed Training**: Multi-GPU support with `torch.distributed.run`
- **Automatic Validation Split**: Creates 10% validation from training data
- **Best Checkpoint Saving**: Saves `checkpoint_best.pth` based on validation loss
- **Mixed Precision**: TF32 enabled for faster training on Ampere GPUs

---

## 🖥️ Using the Plant Diagnostic System

### **Image Analysis Workflow**

#### **Option 1: Webcam Capture (Recommended)**
1. **Enable Webcam**: Click the camera icon in the image upload area
2. **Allow Access**: Grant camera permissions when prompted by your browser
3. **Capture Image**: Click the camera icon again to capture a live photo
4. **Adjust Settings**: Use the temperature slider (0.01-0.5) to control response creativity
5. **Run Analysis**: Click "📤 Send" in the Standard Analysis panel or "🔎 Analyze" in Enhanced Analysis

#### **Option 2: File Upload**
1. **Upload Image**: Click the image upload area and select a clear photo of your strawberry plant
2. **Adjust Settings**: Use the temperature slider (0.01-0.5) to control response creativity
3. **Run Analysis**: Click "📤 Send" in the Standard Analysis panel or "🔎 Analyze" in Enhanced Analysis

### **Understanding the Results**

The system provides structured medical reports with:

**🔍 Diagnosis Section:**
- **7 Disease Classes**: healthy, overwatering, root rot, drought, frost injury, gray mold, white mold
- **Confidence Indicators**: 🟢 High (≥90%), 🟡 Medium (70-90%), 🔴 Low (<70%)

**📋 Medical Report Format:**
```
1) Diagnosis: [Disease Name]
2) Visible cues: [Specific visual observations from the image]
3) Recommendation: [Actionable treatment steps]
```

**🎯 Best Practices:**
- **Webcam Usage**: Use webcam for real-time diagnosis - it's faster and more convenient
- **Image Quality**: Capture clear, well-lit images of affected plant areas
- **Multiple Angles**: Include both close-ups and full plant views when possible
- **Temperature Settings**: Use temperature 0.1-0.3 for balanced creativity and accuracy
- **Enhanced Analysis**: Use Enhanced Analysis for detailed treatment recommendations with web research
- **Knowledge Graph**: Check the Knowledge Graph tab for related agricultural insights

---

## 🧪 How the System Works

### **Two-Stage AI Pipeline**

1. **ResNet-50 Classification**:
   - Pre-trained on ImageNet, fine-tuned on strawberry diseases
   - Uses Test-Time Augmentation (TTA) for robust predictions
   - Temperature scaling for calibrated confidence scores
   - Outputs: `{healthy, overwatering, root rot, drought, frost injury, gray mold, white mold}`

2. **MiniGPT-v2 Explanation**:
   - Vision-language model trained on image-text pairs
   - Receives ResNet diagnosis as "ground truth"
   - Generates detailed medical reports explaining the diagnosis
   - Structured output: diagnosis, visible cues, recommendations

### **Processing Flow**

```
Image Upload → ResNet Classification → Label Mapping → MiniGPT Explanation → Medical Report
```

**Key Features:**
- **Ground Truth Approach**: ResNet diagnosis is treated as absolute truth
- **No Label Drift**: MiniGPT explains, doesn't override the diagnosis
- **Confidence Scoring**: Visual indicators based on ResNet confidence
- **Doctor-Grade Reports**: Structured, professional medical format

---

## 📊 Knowledge Graph

If `kg_nodes_faostat.csv` and `kg_relationships_faostat.csv` are present in the repo root, the **Knowledge Graph** tab renders an interactive Plotly graph. Use **Reload Full Graph** or **Show Crop Neighborhood** to explore.

---

## 📁 Project Structure

```
Plant Diagnostic System/
├── demo_v5.py                           # Main Gradio web interface with webcam support
├── launch_demo_v5.sh                   # Launch script for easy startup
├── resnet_classifier.py                 # ResNet-50 model and inference (symlink)
├── train.py                            # Training script
├── eval_only.py                        # Non-interactive evaluation
├── eval_holdout.py                     # Holdout evaluation script
├── demo-editing/                       # Development tools
│   ├── dev_server.py                   # Hot-reload development server
│   ├── ui_components.py                # Reusable UI components
│   └── HOT_RELOAD_README.md            # Development documentation
├── minigpt4/                           # Core MiniGPT-v2 framework
│   ├── models/                         # Model architectures
│   ├── tasks/                          # Training tasks
│   ├── runners/                        # Training runners
│   └── datasets/                       # Dataset builders
├── eval_configs/                       # Inference configurations
│   └── minigptv2_eval.yaml
├── train_configs/                      # Training configurations
│   └── minigptv2_strawberry_diagnostic.yaml
├── plant_diagnostic/                   # ResNet training and data
│   ├── models/
│   │   └── resnet_straw_final.pth     # 7-class ResNet checkpoint
│   ├── data/                          # Training images
│   └── datasets/                      # Dataset annotations
├── llama_weights/                      # LLaMA-2-7B weights
│   └── Llama-2-7b-chat-hf/
├── output/                            # Training outputs
│   └── minigptv2_strawberry_diagnostic/
├── kg_nodes_faostat.csv               # Knowledge graph nodes
├── kg_relationships_faostat.csv       # Knowledge graph edges
├── dark_theme.css                     # UI styling
└── docs/                              # Documentation
    └── DEMO_V5_VERIFICATION.md        # System verification report
```

---

## 🧩 Technical Details

### **Model Architecture**
- **ResNet-50**: ImageNet pre-trained, fine-tuned on 7-class strawberry dataset
- **MiniGPT-v2**: Vision-language model with LLaMA-2-7B backbone
- **LoRA Fine-tuning**: Efficient adaptation with rank-16 LoRA adapters

### **Performance Optimizations**
- **TF32 Precision**: Enabled for faster training on Ampere GPUs
- **Mixed Precision Training**: Automatic mixed precision with gradient scaling
- **CUDA Memory Management**: Expandable segments for better memory utilization
- **Test-Time Augmentation**: Horizontal flip for robust ResNet predictions

### **Training Features**
- **Automatic Validation Split**: 10% holdout from training data
- **Best Checkpoint Saving**: Saves model with lowest validation loss
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Scheduling**: Cosine annealing with warmup

---

## 🔧 Troubleshooting

### **Common Issues**

#### **Webcam Issues**
- **Camera permissions**: Ensure browser has camera access when prompted
- **Camera in use**: Close other applications using the camera (Zoom, Teams, etc.)
- **Browser compatibility**: Use Chrome/Firefox for best results (Safari also supported)
- **HTTPS required**: Some browsers require HTTPS for camera access (use `--share` flag)
- **Camera not detected**: Check if camera is connected and not being used by other apps

#### **Model Loading Issues**
- **Check paths**: Verify model files exist in expected locations
- **CUDA availability**: Ensure CUDA is properly installed
- **Dependencies**: Install all required packages
- **Memory**: Ensure sufficient GPU memory (8GB+ recommended)

#### **Performance Issues**
- **Slow inference**: Use `--resnet-anchor` flag for faster first inference
- **Memory errors**: Reduce image size or use CPU mode
- **Webcam lag**: Close other applications to free up system resources

---

## 🗺️ Future Enhancements

- **Multi-Crop Support**: Expand beyond strawberries to other crops
- **Advanced Augmentation**: More sophisticated data augmentation strategies
- **Real-time Processing**: Optimize for mobile/edge deployment
- **Confidence Calibration**: Improve uncertainty quantification
- **Export Features**: JSON/CSV export for dataset analysis
- **Docker Deployment**: Containerized deployment option

---

## 📜 License & Credits

This project builds upon:
- **MiniGPT-v2**: Vision-language model framework
- **LLaMA-2**: Language model backbone
- **ResNet**: Image classification architecture
- **FAOSTAT**: Agricultural knowledge graph data

Please respect upstream licenses and dataset terms of use.

---

## 📚 Citation

```bibtex
@software{plant_diagnostic_system,
  title  = {Plant Diagnostic System: AI-Powered Strawberry Disease Detection},
  author = {William Starks, Gus Marcum, Kiriti Vundavilli},
  year   = {2025},
  note   = {GitHub repository: Advanced plant health analysis with ResNet + MiniGPT-v2}
}
```

---

## 🧪 Summary

The Plant Diagnostic System combines ResNet-50 classification with MiniGPT-v2 explanation to provide accurate, doctor-grade strawberry plant health analysis. The system treats ResNet predictions as ground truth and generates structured medical reports with confidence scoring and actionable recommendations.
