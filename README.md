# 🍓 Strawberry Plant Diagnostic (MiniGPT‑v2 + ResNet)

An image-based **strawberry plant diagnostician** that pairs a lightweight **ResNet** classifier (to pin the diagnosis) with **MiniGPT‑v2** (to explain visible cues and next steps). Ships with a dark, two‑pane **Gradio** UI and an optional interactive **FAOSTAT** knowledge graph.

> Research prototype. Not medical/agronomic advice.

---

## ✨ Features

- **Dual engine**: ResNet classifier + MiniGPT‑v2 reasoning (LLM explains; it does **not** override a high‑confidence label)
- **One‑Diagnosis guarantee**: post‑filter enforces **exactly one** `Diagnosis:` line (no duplicates or “Dx:” echoes)
- **Confidence badge** on each result (🟢 ≥0.90, 🟡 ≥0.70, 🔴 else)
- **Two chat panes**: Standard analysis and “Enhanced” (web context optional; can be disabled)
- **Interactive FAOSTAT graph** (Plotly + NetworkX) with full‑graph and neighborhood views
- **Dark, responsive UI**

---

## 🧰 Requirements

- Python **3.9+**
- **CUDA GPU** recommended (MiniGPT‑v2 + LLaMA‑2 7B)
- Model weights (local paths set in your YAML/code):
  - LLaMA‑2‑7B (chat) weights (e.g., `llama_weights/Llama-2-7b-chat-hf`)
  - MiniGPT‑v2 checkpoint (e.g., `output/minigptv2_finetune/new/checkpoint_9.pth`)
  - ResNet weights for the strawberry classifier (e.g., `plant_diagnostic/models/resnet_straw5.pth`)
- (Optional) **SERPAPI** key for the Enhanced pane (diagnostics work without it)

---

## ⚙️ Setup

```bash
git clone https://github.com/adainstarks/strawberry-diagnostic.git
cd MiniGPT-4

# Pick the CUDA wheel matching your system:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# App/runtime deps
pip install transformers==4.41.1 bitsandbytes gradio pillow numpy pandas plotly networkx python-dotenv
```

(Optional) create `.env` if you plan to use the Enhanced pane:

```env
SERP_API_KEY=your_serpapi_key_here
```

> If you don’t use Enhanced, you can skip the key.

---

## 🔧 Configure

Point the config to your local model paths in `configs/models/minigpt_v2.yaml`:

```yaml
model:
  arch: minigpt_v2
  model_type: pretrain
  max_txt_len: 256
  ckpt: /MiniGPT-4/output/minigptv2_finetune/new/checkpoint_9.pth
  llama_model: /MiniGPT-4/llama_weights/Llama-2-7b-chat-hf
  use_lora: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  tune_mm_mlp_adapter: true
  freeze_llama: true

preprocess:
  vis_processor:
    eval: { name: blip2_image_eval, image_size: 576 }

run:
  task: image_text_pretrain
```

Ensure your ResNet is loaded in code, e.g.:

```python
# resnet_classifier.py or where you call it
load_resnet("plant_diagnostic/models/resnet_straw5.pth")
```

---

## 🚀 Run

This is the exact dev command used:

```bash
python demo_v5.py --cfg-path configs/models/minigpt_v2.yaml --gpu-id 0
```

Preload the ResNet on launch (minor first‑image speedup):

```bash
python demo_v5.py --cfg-path configs/models/minigpt_v2.yaml --gpu-id 0 --resnet-anchor
```

Gradio will print a local URL to open in your browser.

---

## 🖥️ Using the app

1. **Upload** a close‑up strawberry photo (leaf/flower/fruit). 
2. Keep **Temperature** near **0.05–0.10** for stable, grounded output. 
3. Read the **Standard Analysis** panel:
   - `Diagnosis: <Healthy / Unknown / Frost injury / Root rot / Overwatering / Drought>`
   - `Visible cues: …` (2–4 image‑grounded cues tied to locations: leaf margins, fruit surface, crown)
   - `Step: …` (one immediate, practical action)

---

## 🧪 How it works (short)

1. **ResNet** classifies the uploaded image into `{healthy, drought, overwatering, root rot, frost injury}` with per‑class thresholds. 
2. If confidence is high, that label is **fixed**; the LLM **only** describes visible cues + one actionable step (no label drift). 
3. If confidence is low, the system uses a cautious prompt that avoids disease names and focuses on evidence. 
4. A compact answer is produced; a **confidence badge** is shown when available. 
5. Optional **Enhanced** pane can append brief web context (requires SERP key).
<img width="1024" height="545" alt="Screenshot 2025-09-28 223916" src="https://github.com/user-attachments/assets/decacaea-d8dc-4820-b986-75183cdbf9db" />
<img width="980" height="469" alt="Screenshot 2025-09-28 223816" src="https://github.com/user-attachments/assets/cf6e3cf5-0509-4fb5-90f1-fa987f476e68" />
<img width="1056" height="734" alt="Screenshot 2025-09-28 223732" src="https://github.com/user-attachments/assets/0eec2277-e38e-47de-9069-a5881e86e2bb" />
---

## 📊 Knowledge Graph

If `kg_nodes_faostat.csv` and `kg_relationships_faostat.csv` are present in the repo root, the **Knowledge Graph** tab renders an interactive Plotly graph. Use **Reload Full Graph** or **Show Crop Neighborhood** to explore.

---

## 📁 Project layout (high level)

```
minigpt4/             # core model + conversation stack
configs/models/       # minigpt_v2.yaml (runtime config you use at launch)
eval_configs/         # legacy/optional eval configs
demo_v5.py            # main Gradio app (v5 UI, ResNet+LLM flow)
resnet_classifier.py  # ResNet load + inference helpers
plant_diagnostic/     # ResNet + data helpers
  └─ models/resnet_straw5.pth
kg_nodes_faostat.csv
kg_relationships_faostat.csv
```

---

## 🧩 Dev notes

- Current model state has minor hallucination issues when describing visual symptoms, an increase in training data would help.
- `blip2_image_eval` at **576 px** is a good trade‑off for small lesions vs. throughput. 
- Output cleaning removes control tokens, duplicate sentences, and normalizes whitespace.

---

## 🗺️ Roadmap

- Multi‑crop / center‑crop inference for tiny lesions 
- Optional per‑part cues (leaf/fruit/flower) 
- Export to JSONL/CSV for dataset triage 
- Docker packaging for deployment

---

## 📜 License & credits

- Builds on MiniGPT‑v2 / MiniGPT‑4 and LLaMA‑2.
- Add your own **LICENSE** file if you plan to distribute your modifications.

---

## 📚 Citation

```bibtex
@software{strawberry_minigpt_demo,
  title  = {Strawberry Plant Diagnostic Demo (MiniGPT-v2 + ResNet)},
  author = {Starks, William and collaborators},
  year   = {2025},
  note   = {GitHub repository}
}
```

---

## 🧪 TL;DR

A guarded MiniGPT‑v2 + ResNet pipeline that returns a short, grounded plant diagnosis with one clean `Diagnosis:` line, visible cues, and a practical next step—without naming diseases at low confidence.
