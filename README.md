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

### **Demo/Inference Mode**

Launch the Plant Diagnostic System:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python demo_v5.py --cfg-path eval_configs/minigptv2_eval.yaml --resnet-anchor
```

**Parameters:**
- `--cfg-path`: Points to the evaluation configuration
- `--resnet-anchor`: Preloads ResNet for faster first inference
- `CUDA_VISIBLE_DEVICES=0`: Uses GPU 0 (adjust as needed)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: Optimizes CUDA memory allocation

The system will start a Gradio web interface. Open the provided URL in your browser.

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
- Upload clear, well-lit images of affected plant areas
- Include both close-ups and full plant views when possible
- Use temperature 0.1-0.3 for balanced creativity and accuracy
- Check the Knowledge Graph tab for related agricultural insights

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

<img width="3840" height="804" alt="Untitled diagram _ Mermaid Chart-2025-10-04-233422" src="https://github.com/user-attachments/assets/b6f433ad-d25a-4b06-b0c5-e00addd43984" />
<img width="3840" height="1730" alt="Untitled diagram _ Mermaid Chart-2025-10-04-233702" src="https://github.com/user-attachments/assets/7bf297ba-d818-48be-a803-d94d6994c62c" />
<img width="3840" height="513" alt="Untitled diagram _ Mermaid Chart-2025-10-04-233939" src="https://github.com/user-attachments/assets/fc1dfc01-c445-4c3f-9398-709c4845fcfc" />

```
Plant Diagnostic System/
├── demo_v5.py                           # Main Gradio web interface
├── resnet_classifier.py                 # ResNet-50 model and inference
├── train.py                            # Training script
├── eval_only.py                        # Non-interactive evaluation
├── eval_holdout.py                     # Holdout evaluation script
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
└── dark_theme.css                     # UI styling
```

---

## 🧩 Technical Details

### **Model Architecture**
- **ResNet-50**: ImageNet pre-trained, fine-tuned on dataset
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

## 🗺️ Future Enhancements

- **Multi-Crop Support**: Expand beyond strawberries to other crops
- **Advanced Augmentation**: More sophisticated data augmentation strategies

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
  author = {William Starks, Kiriti Vundavilli},
  year   = {2025},
  note   = {GitHub repository: Advanced plant health analysis with ResNet + MiniGPT-v2}
}
```

---

## 🧪 Summary

The Plant Diagnostic System combines ResNet-50 classification with MiniGPT-v2 explanation to provide accurate, doctor-grade strawberry plant health analysis. The system treats ResNet predictions as ground truth and generates structured medical reports with confidence scoring and actionable recommendations.
