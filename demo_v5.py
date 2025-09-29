#William Starks - Plant Diagnostic MiniGPT, derived from demo_v4.py and modified into a resnet50-wired strawberry pathologist. WIP
#Added CSS to the gradio app for user satisfaction
import argparse
import os
import re
import numpy as np
from PIL import Image
import torch
import gradio as gr
from serpapi import GoogleSearch
import torch.backends.cudnn as cudnn
import networkx as nx
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from functools import lru_cache
import logging
from datetime import datetime

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# Import modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from resnet_classifier import load_resnet, diagnose_or_none

# Configure logging
logging.basicConfig(filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.ERROR)

# SERPAPI Configuration
SERP_API_KEY = 'fb6c393ddf8c395e92e9f5e4244f1e3ee75084f85c94aabde8954cd12c5df1ea'
if not SERP_API_KEY:
    raise ValueError("SERP_API_KEY environment variable not set.")

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="MiniGPT-v2 Demo")
    parser.add_argument("--cfg-path", default="eval_configs/minigptv2_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--options", default=None, help="additional options to override the configuration.")
    parser.add_argument("--resnet-anchor", action="store_true",
                    help="Anchor first user turn with ResNet diagnosis when confident")
    args = parser.parse_args()
    return args

# --- ResNet anchor helpers ---
_RESNET_MODEL = None

def _get_resnet():
    global _RESNET_MODEL
    if _RESNET_MODEL is None:
        _RESNET_MODEL = load_resnet("plant_diagnostic/models/resnet_straw5.pth")
    return _RESNET_MODEL

args = parse_args()
if args.resnet_anchor:
    _ = _get_resnet()

# --- Fixed-label helpers ---
_CLASS_THRESH = {
    "healthy":      0.67,
    "overwatered":  0.74,
    "root_rot":     0.76,
    "drought":      0.77,
    "frost":        0.79,
}

_CANON_LABEL_MAP = {
    "healthy": "healthy",
    "overwatered": "overwatering",
    "root_rot": "root rot",
    "drought": "drought",
    "frost": "frost injury",
}

def _accept_label(pred) -> str:
    if not pred:
        return "unknown"
    lbl = str(pred.get("label", "")).lower().strip()
    p1  = float(pred.get("p1", 0.0))
    if lbl in _CLASS_THRESH and p1 >= _CLASS_THRESH[lbl]:
        return _CANON_LABEL_MAP[lbl]
    return "unknown"

def _postprocess_caption(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = t.replace(""", "").replace(""", "").replace("'", "").replace('"', "")
    t = re.sub(r'</?[^>\s]{1,32}>?', '', t)
    t = t.replace('\u200b', '').replace('<', '').replace('>', '').replace('\\n', ' ')
    parts = re.split(r'(?<=[.!?])\s+', t)
    cleaned, seen = [], set()
    for s in parts:
        s = s.strip()
        if not s:
            continue
        if s.lower().startswith("in conclusion"):
            s = s.split(":", 1)[-1].strip() or s
        key = re.sub(r"\s+", " ", s.lower())
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s)
    text = " ".join(cleaned)
    words = text.split()
    if len(words) > 140:
        text = " ".join(words[:140])
    return text

# Load configuration
cfg = Config(args)
if args.options is not None:
    cfg.update_with_str(args.options)

# Set device
device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_id)

# Initialize model
model_config = cfg.model_cfg
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
model.load_checkpoint(cfg.model_cfg.ckpt)
model = model.eval()
print(f"[ckpt] using: {cfg.model_cfg.ckpt}")

# Model configuration patches
try:
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
        try:
            model.generation_config.cache_implementation = "static"
        except Exception:
            pass
except Exception:
    pass

try:
    if hasattr(model, "config"):
        model.config.use_cache = False
except Exception:
    pass

def _patch_drop_cachepos_all(root):
    wrapped = 0
    for module in root.modules():
        f = getattr(module, "forward", None)
        if f is None:
            continue
        if getattr(f, "_drops_cachepos", False):
            continue

        def wrapped_forward(*args, __orig=f, **kwargs):
            kwargs.pop("cache_position", None)
            return __orig(*args, **kwargs)

        setattr(wrapped_forward, "_drops_cachepos", True)
        try:
            module.forward = wrapped_forward
            wrapped += 1
        except Exception:
            pass
    print(f"[patch] drop(cache_position): wrapped {wrapped} module.forward funcs")

_patch_drop_cachepos_all(model)

# Initialize visual processor
try:
    vis_processor_cfg = cfg.preprocess_cfg.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
except Exception:
    vis_processor = registry.get_processor_class('blip2_image_eval').from_config({'image_size': 448})

# Chat patches
try:
    _ORIG_ANSWER_PREPARE = Chat.answer_prepare
    def _answer_prepare_nocache(self, *args, **kwargs):
        kwargs.pop("use_cache", None)
        return _ORIG_ANSWER_PREPARE(self, *args, **kwargs)
    Chat.answer_prepare = _answer_prepare_nocache
    print("[patch] Chat.answer_prepare patched to drop 'use_cache'")
except Exception as e:
    print(f"[patch] Could not patch answer_prepare: {e}")

# Initialize chat
chat = Chat(model, vis_processor, device=device)
print('Initialization Finished')

# Define CONV_VISION
CONV_VISION = Conversation(
    system="",
    roles=(r"<s>[INST] ", r" [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

def fetch_serp_context(query):
    """Fetch additional context from SERPAPI."""
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERP_API_KEY,
        "num": 3,
        "gl": "us",
        "hl": "en",
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results", [])
        context = " ".join([result.get("snippet", "") for result in organic_results[:3]])
        return context
    except Exception as e:
        logging.error(f"SERPAPI Error: {e}")
        return ""

def _empty_fig(msg):
    """Create an empty figure with a message."""
    fig = go.Figure()
    fig.update_layout(
        annotations=[dict(
            text=msg, 
            x=0.5, 
            y=0.5, 
            showarrow=False,
            font=dict(color="#ffffff", size=14)
        )],
        width=1200, 
        height=700,
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor='rgba(17, 17, 27, 0.9)',
        plot_bgcolor='rgba(17, 17, 27, 0.9)',
    )
    return fig

@lru_cache(maxsize=1)
def _load_csvs():
    """Load and cache CSV files."""
    try:
        n = pd.read_csv('kg_nodes_faostat.csv', low_memory=False)
        r = pd.read_csv('kg_relationships_faostat.csv', low_memory=False)
        n['id'] = n['id'].astype(str).str.strip()
        r['start_id'] = r['start_id'].astype(str).str.strip()
        r['end_id'] = r['end_id'].astype(str).str.strip()
        return n, r
    except FileNotFoundError as e:
        print(f"CSV files not found: {e}")
        return pd.DataFrame(), pd.DataFrame()

def create_knowledge_graph(
    nodes_path='kg_nodes_faostat.csv',
    relationships_path='kg_relationships_faostat.csv',
    max_edges=2000
):
    """Create the main knowledge graph visualization with dark theme."""
    try:
        base = Path(__file__).resolve().parent
        nodes_fp = Path(nodes_path)
        rels_fp = Path(relationships_path)
        if not nodes_fp.exists(): nodes_fp = base / nodes_path
        if not rels_fp.exists(): rels_fp = base / relationships_path

        nodes_df, relationships_df = _load_csvs()
        
        if nodes_df.empty or relationships_df.empty:
            return _empty_fig("CSV files not found or empty")

        need_nodes = {'id','label','name'}
        need_rels = {'start_id','end_id','type'}
        if not need_nodes.issubset(nodes_df.columns):
            return _empty_fig(f"Missing node columns: {need_nodes - set(nodes_df.columns)}")
        if not need_rels.issubset(relationships_df.columns):
            return _empty_fig(f"Missing rel columns: {need_rels - set(relationships_df.columns)}")

        idset = set(nodes_df['id'])
        relationships_df = relationships_df[
            relationships_df['start_id'].isin(idset) & relationships_df['end_id'].isin(idset)
        ]

        if len(relationships_df) > max_edges:
            relationships_df = relationships_df.sample(n=max_edges, random_state=42)

        G = nx.DiGraph()
        for _, row in nodes_df.iterrows():
            G.add_node(row['id'], label=row['label'], name=row['name'])
        for _, row in relationships_df.iterrows():
            G.add_edge(row['start_id'], row['end_id'], type=row['type'])

        if G.number_of_nodes() == 0:
            return _empty_fig("No nodes to display.")
        
        k = 1.5 / max(1, G.number_of_nodes() ** 0.5)
        pos = nx.spring_layout(G, k=k, iterations=200, seed=42)

        # Enhanced color scheme for dark theme
        def _edge_trace_for(rel_type, color):
            ex, ey, et = [], [], []
            for u, v, d in G.edges(data=True):
                if d.get('type') != rel_type:
                    continue
                x0, y0 = pos[u]; x1, y1 = pos[v]
                ex.extend([x0, x1, None])
                ey.extend([y0, y1, None])
                t = d.get('type', '')
                et.extend([t, t, None])
            return go.Scatter(
                x=ex, y=ey,
                line=dict(width=0.8, color=color),
                hoverinfo='text',
                text=et,
                mode='lines',
                opacity=0.6
            )

        edge_trace_measured = _edge_trace_for('Measured_By', '#4a5568')
        edge_trace_thrives = _edge_trace_for('Thrives_In', '#00d4ff')
        edge_trace_hasimg = _edge_trace_for('Has_Image', '#b794f4')
        edge_trace_hascond = _edge_trace_for('Has_Condition', '#ff6b6b')
        edge_trace_hasrsn = _edge_trace_for('Has_Reason', '#4ecdc4')

        label_colors = {
            'Crop': '#00d4ff',
            'Region': '#48bb78',
            'Element': '#ff6b6b',
            'Condition': '#ffd93d',
            'Image': '#b794f4',
            'Reason': '#4ecdc4'
        }

        node_x, node_y, node_text, node_color = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            attrs = G.nodes[node]
            node_text.append(f"<b>{attrs.get('label','?')}</b><br>{attrs.get('name','?')}")
            node_color.append(label_colors.get(attrs.get('label'), '#718096'))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=12,
                color=node_color,
                line=dict(width=2, color='rgba(255, 255, 255, 0.3)'),
            ),
            textposition="top center",
            textfont=dict(size=9, color='rgba(255, 255, 255, 0.7)')
        )

        fig = go.Figure(
            data=[edge_trace_measured, edge_trace_thrives, edge_trace_hasimg, 
                  edge_trace_hascond, edge_trace_hasrsn, node_trace],
            layout=go.Layout(
                title=dict(
                    text='<b>FAOSTAT Knowledge Graph</b>',
                    font=dict(size=20, color='#ffffff'),
                    x=0.5,
                    xanchor='center'
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=60),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=1200,
                height=700,
                paper_bgcolor='rgba(17, 17, 27, 0.9)',
                plot_bgcolor='rgba(17, 17, 27, 0.9)',
                hoverlabel=dict(
                    bgcolor="rgba(30, 30, 45, 0.95)",
                    font_size=12,
                    font_family="monospace",
                    font_color="white"
                )
            )
        )
        return fig

    except Exception as e:
        print(f"[KG ERROR] {e}")
        return _empty_fig(f"Error: {e}")

def draw_crop_from_csv(c_id):
    """Draw the neighborhood graph for a specific crop ID with dark theme."""
    try:
        nodes_df, rels_df = _load_csvs()
        
        if nodes_df.empty or rels_df.empty:
            return _empty_fig("CSV files not found or empty")
        
        nbr = rels_df[(rels_df['start_id'] == c_id) | (rels_df['end_id'] == c_id)]
        if nbr.empty:
            return _empty_fig(f"No edges for crop id {c_id}")

        keep_ids = set([c_id]) | set(nbr['start_id']) | set(nbr['end_id'])
        sub_nodes = nodes_df[nodes_df['id'].isin(keep_ids)]

        G = nx.DiGraph()
        for _, row in sub_nodes.iterrows():
            G.add_node(row['id'], label=row['label'], name=row['name'])
        for _, row in nbr.iterrows():
            G.add_edge(row['start_id'], row['end_id'], type=row['type'])

        k = 1.5 / max(1, G.number_of_nodes() ** 0.5)
        pos = nx.spring_layout(G, k=k, iterations=200, seed=42)

        label_colors = {
            'Crop': '#00d4ff',
            'Region': '#48bb78',
            'Element': '#ff6b6b',
            'Condition': '#ffd93d',
            'Image': '#b794f4',
            'Reason': '#4ecdc4'
        }

        def _edge_trace_for(rel_type, color):
            ex, ey, et = [], [], []
            for u, v, d in G.edges(data=True):
                if d.get('type') != rel_type:
                    continue
                x0, y0 = pos[u]; x1, y1 = pos[v]
                ex.extend([x0, x1, None])
                ey.extend([y0, y1, None])
                t = d.get('type','')
                et.extend([t, t, None])
            return go.Scatter(
                x=ex, y=ey, 
                line=dict(width=1.2, color=color),
                hoverinfo='text', 
                text=et, 
                mode='lines',
                opacity=0.7
            )

        edge_trace_measured = _edge_trace_for('Measured_By', '#4a5568')
        edge_trace_thrives = _edge_trace_for('Thrives_In', '#00d4ff')
        edge_trace_hasimg = _edge_trace_for('Has_Image', '#b794f4')
        edge_trace_hascond = _edge_trace_for('Has_Condition', '#ff6b6b')
        edge_trace_hasrsn = _edge_trace_for('Has_Reason', '#4ecdc4')

        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        for n in G.nodes():
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)
            a = G.nodes[n]
            node_text.append(f"<b>{a.get('label','?')}</b><br>{a.get('name','?')}")
            node_color.append(label_colors.get(a.get('label'), '#718096'))
            node_size.append(20 if n == c_id else 12)

        node_trace = go.Scatter(
            x=node_x, y=node_y, 
            mode='markers',
            hoverinfo='text', 
            text=node_text,
            marker=dict(
                size=node_size, 
                color=node_color,
                line=dict(width=2, color='rgba(255, 255, 255, 0.3)')
            )
        )

        return go.Figure(
            data=[edge_trace_measured, edge_trace_thrives, edge_trace_hasimg,
                  edge_trace_hascond, edge_trace_hasrsn, node_trace],
            layout=go.Layout(
                title=dict(
                    text=f'<b>Neighborhood: {c_id}</b>',
                    font=dict(size=20, color='#ffffff'),
                    x=0.5,
                    xanchor='center'
                ),
                showlegend=False, 
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=60),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=1200, 
                height=700,
                paper_bgcolor='rgba(17, 17, 27, 0.9)',
                plot_bgcolor='rgba(17, 17, 27, 0.9)',
                hoverlabel=dict(
                    bgcolor="rgba(30, 30, 45, 0.95)",
                    font_size=12,
                    font_family="monospace",
                    font_color="white"
                )
            )
        )
    except Exception as e:
        print(f"[Draw Crop ERROR] {e}")
        return _empty_fig(f"Error: {e}")

def process_chat_with_image(user_message, chatbot, chat_state, gr_img, img_list, temperature, is_enhanced=False):
    """Process chat with image analysis (minimal gating; two fixed prompts)."""
    try:
        if gr_img is None:
            return (chatbot + [[user_message, "⚠️ Please upload an image first."]], chat_state, img_list)

        # Fresh conversation + encode image
        chat_state = CONV_VISION.copy()
        img_list = []
        chat.upload_img(gr_img, chat_state, img_list)
        chat.encode_img(img_list)

        # Lightweight ResNet pass (just to get a label + confidence)
        img_path = "/tmp/tmp_image.jpg"
        gr_img.save(img_path)
        pred = None
        try:
            pred = diagnose_or_none(_get_resnet(), img_path, img_size=256)
        except Exception as e:
            print(f"[resnet] warn: {e}")

        # Accept/canonize label once; no extra thresholds here
        final_label = "unknown"
        try:
            final_label = _accept_label(pred)  # returns e.g., {"healthy","frost injury","root rot","overwatering","drought","unknown"}
        except Exception:
            pass

        # Confidence (optional badge)
        try:
            p1 = float(pred.get("p1", 0.0)) if pred else 0.0
        except Exception:
            p1 = 0.0

        # ---------------------------
        # Two fixed system prompts
        # ---------------------------
        # NOTE: Model was trained on <image> only; keep prompts simple and deterministic.
        if str(final_label).lower().strip() == "healthy":
            chat_state.system = (
                "<<SYS>>You are a plant diagnostician. Look only at the photo.\n"
                "Output exactly THREE sentences, in this exact format:\n"
                "1) Diagnosis: Healthy.\n"
                "2) Visible cues: 2–3 positive cues that are clearly visible (e.g., uniform green foliage, intact margins, firm fruit; avoid technical jargon).\n"
                "3) Step: one simple maintenance action (light watering cadence, debris removal, light mulch). No lists.\n"
                "Do not invent diseases or tools. Do not add extra sentences or disclaimers.\n"
                "<</SYS>>"
            )
        else:
            # Non-healthy OR unknown → still say one diagnosis line and proceed
            # (We pass the fixed diagnosis text directly to keep it single and consistent.)
            chat_state.system = (
                "<<SYS>>You are a plant diagnostician. Look only at the photo.\n"
                "Output exactly THREE sentences, in this exact format:\n"
                f"1) Diagnosis: {final_label.title()}.\n"
                "2) Visible cues: 2–4 clearly visible cues tied to locations (color, margins, curl/wilt, spots/lesions, translucency, crown/soil appearance). No guessing.\n"
                "3) Step: one immediate, practical action; include a numeric recovery window like '3–7 days' if applicable; otherwise say why not.\n"
                "Do not add extra sentences, lists, sensors, or professional referrals.\n"
                "<</SYS>>"
            )

        # Minimal user ask text — keep deterministic
        ask_text = (
            user_message.strip()
            if user_message and user_message.strip()
            else "Produce the three-sentence output now."
        )
        _ = chat.ask(ask_text, chat_state)

        ans = chat.answer(
            conv=chat_state,
            img_list=img_list,
            temperature=0.05,
            max_new_tokens=300,
            max_length=900,
        )
        body = (ans[0] if isinstance(ans, (list, tuple)) and len(ans) else str(ans)).strip()

        # Light cleanup only (no aggressive rewrites)
        body = _postprocess_caption(body)

        # Optional confidence badge prefix
        if p1 > 0:
            badge = "🟢" if p1 >= 0.90 else "🟡" if p1 >= 0.70 else "🔴"
            body = f"{badge} **Confidence: {p1:.1%}**\n\n{body}"

        return (chatbot + [[user_message, body]], chat_state, img_list)

    except Exception as e:
        logging.error(f"Error in chat processing: {str(e)}")
        return (chatbot + [[user_message, f"❌ Error: {str(e)}"]], chat_state, img_list)



def reset_chat(chat_state, img_list):
    """Reset chat state and image list."""
    return [], CONV_VISION.copy(), []

# Load custom CSS from external file
def load_custom_css():
    """Load the dark theme CSS from external file."""
    css_file_path = Path(__file__).resolve().parent / "dark_theme.css"
    
    # Fallback CSS if file not found
    fallback_css = """
    .gradio-container {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e0e0e0 !important;
    }
    """
    
    try:
        if css_file_path.exists():
            with open(css_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            print(f"[WARNING] CSS file not found at {css_file_path}. Using fallback CSS.")
            return fallback_css
    except Exception as e:
        print(f"[ERROR] Failed to load CSS: {e}. Using fallback CSS.")
        return fallback_css

# Load the CSS
custom_css = load_custom_css()

# Create the Gradio interface
with gr.Blocks(
    title="🌿 Plant Diagnostic System",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.cyan,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        # Force dark backgrounds globally
        body_background_fill="linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%)",
        body_background_fill_dark="linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%)",
        background_fill_primary="rgba(20, 20, 35, 0.9)",
        background_fill_primary_dark="rgba(20, 20, 35, 0.9)",
        background_fill_secondary="rgba(25, 25, 40, 0.8)",
        background_fill_secondary_dark="rgba(25, 25, 40, 0.8)",
        border_color_primary="rgba(0, 212, 255, 0.2)",
        border_color_primary_dark="rgba(0, 212, 255, 0.2)",
        # Text colors
        body_text_color="#e0e0e0",
        body_text_color_dark="#e0e0e0",
        body_text_color_subdued="#a0a0a0",
        body_text_color_subdued_dark="#a0a0a0",
        # Component colors
        color_accent_soft="rgba(0, 212, 255, 0.1)",
        color_accent_soft_dark="rgba(0, 212, 255, 0.1)",
        # Block styling
        block_background_fill="rgba(20, 20, 35, 0.7)",
        block_background_fill_dark="rgba(20, 20, 35, 0.7)",
        block_border_color="rgba(0, 212, 255, 0.15)",
        block_border_color_dark="rgba(0, 212, 255, 0.15)",
        block_label_background_fill="transparent",
        block_label_background_fill_dark="transparent",
        # Input styling
        input_background_fill="rgba(10, 10, 20, 0.8)",
        input_background_fill_dark="rgba(10, 10, 20, 0.8)",
        input_border_color="rgba(0, 212, 255, 0.2)",
        input_border_color_dark="rgba(0, 212, 255, 0.2)",
        input_border_color_focus="rgba(0, 212, 255, 0.5)",
        input_border_color_focus_dark="rgba(0, 212, 255, 0.5)",
    ),
    css=custom_css
) as demo:
    
    # Header with animated gradient
    gr.HTML("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="font-size: 3em; margin-bottom: 10px;">
                🌿 Plant Diagnostic System
            </h1>
            <p style="color: #a0a0a0; font-size: 1.2em;">
                Advanced AI-Powered Strawberry Plant Health Analysis
            </p>
            <div style="display: flex; justify-content: center; gap: 20px; margin-top: 20px;">
                <span class="status-badge status-healthy">✓ System Online</span>
                <span class="status-badge" style="background: rgba(102, 126, 234, 0.8); color: white;">
                    🔬 ResNet + MiniGPT-v2
                </span>
                <span class="status-badge" style="background: rgba(183, 148, 244, 0.8); color: white;">
                    📊 Knowledge Graph Ready
                </span>
            </div>
        </div>
    """)
    
    # Main tabs
    with gr.Tabs() as tabs:
        # Image Analysis Tab
        with gr.Tab("🔬 Image Analysis", elem_classes="custom-tab"):
            with gr.Row():
                # Left column - Image upload and controls
                with gr.Column(scale=1):
                    gr.Markdown("### 📸 Upload Image")
                    image = gr.Image(
                        type="pil", 
                        label="Plant Image",
                        elem_classes="image-upload"
                    )
                    
                    # Analysis settings card
                    with gr.Group():
                        gr.Markdown("### ⚙️ Analysis Settings")
                        temperature = gr.Slider(
                            minimum=0.01,
                            maximum=0.3,
                            value=0.05,
                            step=0.01,
                            label="🌡️ Temperature",
                            info="Lower = More focused | Higher = More creative"
                        )
                        
                        # Analysis status
                        gr.HTML("""
                            <div style="padding: 10px; background: rgba(0, 212, 255, 0.1); 
                                        border-radius: 8px; border: 1px solid rgba(0, 212, 255, 0.3);">
                                <p style="margin: 0; color: #00d4ff; font-size: 0.9em;">
                                    💡 Tip: Upload a clear image of your strawberry plant for best results
                                </p>
                            </div>
                        """)
                        
                    clear = gr.Button(
                        "🔄 Reset All", 
                        variant="stop",
                        size="lg"
                    )

                # Right column - Chat interfaces
                with gr.Column(scale=2):
                    # Standard Analysis
                    with gr.Group():
                        gr.Markdown("### 🤖 Standard Analysis")
                        gr.Markdown("*Quick diagnosis based on visual inspection*")
                        
                        basic_chatbot = gr.Chatbot(
                            height=300,
                            bubble_full_width=False,
                            avatar_images=["🧑‍🌾", "🤖"]
                        )
                        basic_chat_state = gr.State(CONV_VISION.copy())
                        basic_img_list = gr.State([])
                        
                        with gr.Row():
                            basic_input = gr.Textbox(
                                placeholder="Ask about the plant's condition...",
                                scale=4,
                                label=None,
                                container=False
                            )
                            basic_send = gr.Button(
                                "📤 Send", 
                                scale=1, 
                                variant="primary"
                            )

                    # Enhanced Analysis
                    with gr.Group():
                        gr.Markdown("### 🔍 Enhanced Analysis")
                        gr.Markdown("*Comprehensive diagnosis with web research*")
                        
                        enhanced_chatbot = gr.Chatbot(
                            height=300,
                            bubble_full_width=False,
                            avatar_images=["🧑‍🌾", "🔬"]
                        )
                        enhanced_chat_state = gr.State(CONV_VISION.copy())
                        enhanced_img_list = gr.State([])
                        
                        with gr.Row():
                            enhanced_input = gr.Textbox(
                                placeholder="Get detailed analysis with treatment recommendations...",
                                scale=4,
                                label=None,
                                container=False
                            )
                            enhanced_send = gr.Button(
                                "🔎 Analyze", 
                                scale=1, 
                                variant="primary"
                            )

        # Knowledge Graph Tab
        with gr.Tab("📊 Knowledge Graph", elem_classes="custom-tab"):
            gr.Markdown("""
                ### 🌐 FAOSTAT Agricultural Knowledge Network
                <span style="color: #a0a0a0;">Explore relationships between crops, conditions, and agricultural data</span>
            """)
            
            # Graph display with full width
            graph_plot = gr.Plot(label=None)
            
            # Controls below the graph
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### 🎛️ Graph Controls")
                        reload_graph = gr.Button(
                            "🔄 Reload Full Graph",
                            variant="primary",
                            size="lg"
                        )
                
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### 🎯 Crop Explorer")
                        crop_id = gr.Textbox(
                            label="Crop ID",
                            placeholder="e.g., 144 (strawberries)",
                            info="View specific crop neighborhood"
                        )
                        show_btn = gr.Button(
                            "🔍 Show Neighborhood",
                            variant="secondary",
                            size="lg"
                        )
                
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### ⚡ Quick Access")
                        gr.Button("🍓 Strawberries", size="sm").click(
                            lambda: "144", None, crop_id
                        ).then(
                            draw_crop_from_csv, inputs=[crop_id], outputs=[graph_plot]
                        )
                        gr.Button("🍅 Tomatoes", size="sm").click(
                            lambda: "388", None, crop_id
                        ).then(
                            draw_crop_from_csv, inputs=[crop_id], outputs=[graph_plot]
                        )

        # About Tab
        with gr.Tab("ℹ️ About", elem_classes="custom-tab"):
            gr.HTML("""
            <div style="padding: 20px; color: #e0e0e0;">
                <h2 style="color: #00d4ff; margin-bottom: 20px;">🌿 Plant Diagnostic System v2.0</h2>
                
                <div style="background: rgba(25, 25, 40, 0.8); padding: 20px; border-radius: 12px; margin-bottom: 20px;">
                    <h3 style="color: #b794f4; margin-bottom: 15px;">🎯 Features</h3>
                    <ul style="color: #d0d0d0; line-height: 1.8;">
                        <li><strong style="color: #00d4ff;">Dual AI Analysis:</strong> ResNet classifier + MiniGPT-v2 vision model</li>
                        <li><strong style="color: #00d4ff;">Web Integration:</strong> Real-time information from SERPAPI</li>
                        <li><strong style="color: #00d4ff;">Knowledge Graph:</strong> Interactive FAOSTAT agricultural data visualization</li>
                        <li><strong style="color: #00d4ff;">Modern UI:</strong> Dark theme with responsive design</li>
                    </ul>
                </div>
                
                <div style="background: rgba(25, 25, 40, 0.8); padding: 20px; border-radius: 12px; margin-bottom: 20px;">
                    <h3 style="color: #b794f4; margin-bottom: 15px;">🛠️ Technology Stack</h3>
                    <ul style="color: #d0d0d0; line-height: 1.8;">
                        <li><strong style="color: #48bb78;">Vision Models:</strong> ResNet (classification) + MiniGPT-v2 (description)</li>
                        <li><strong style="color: #48bb78;">Data:</strong> FAOSTAT agricultural database</li>
                        <li><strong style="color: #48bb78;">Visualization:</strong> Plotly + NetworkX</li>
                        <li><strong style="color: #48bb78;">Framework:</strong> Gradio with custom theming</li>
                    </ul>
                </div>
                
                <div style="background: rgba(25, 25, 40, 0.8); padding: 20px; border-radius: 12px; margin-bottom: 20px;">
                    <h3 style="color: #b794f4; margin-bottom: 15px;">📈 Confidence Indicators</h3>
                    <div style="display: flex; flex-direction: column; gap: 10px; color: #d0d0d0;">
                        <div><span style="font-size: 1.2em;">🟢</span> <strong style="color: #48bb78;">High Confidence</strong> (>90%): Highly reliable diagnosis</div>
                        <div><span style="font-size: 1.2em;">🟡</span> <strong style="color: #ffd93d;">Medium Confidence</strong> (70-90%): Good reliability</div>
                        <div><span style="font-size: 1.2em;">🔴</span> <strong style="color: #ff6b6b;">Low Confidence</strong> (<70%): Further inspection recommended</div>
                    </div>
                </div>
                
                <div style="background: rgba(25, 25, 40, 0.8); padding: 20px; border-radius: 12px; margin-bottom: 20px;">
                    <h3 style="color: #b794f4; margin-bottom: 15px;">🔍 Best Practices</h3>
                    <ol style="color: #d0d0d0; line-height: 1.8;">
                        <li>Upload clear, well-lit images of affected plant areas</li>
                        <li>Include both close-ups and full plant views when possible</li>
                        <li>Use Enhanced Analysis for detailed treatment recommendations</li>
                        <li>Explore the Knowledge Graph for related agricultural insights</li>
                    </ol>
                </div>
                
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(183, 148, 244, 0.1)); border-radius: 12px; border: 1px solid rgba(0, 212, 255, 0.3);">
                    <p style="color: #00d4ff; margin: 0; font-size: 1.1em;">
                        ✅ System Status: All models loaded and operational
                    </p>
                </div>
            </div>
            """)


    # Event handlers
    basic_send.click(
        process_chat_with_image,
        inputs=[basic_input, basic_chatbot, basic_chat_state, image, basic_img_list, temperature],
        outputs=[basic_chatbot, basic_chat_state, basic_img_list]
    ).then(lambda: "", None, basic_input)
    
    basic_input.submit(
        process_chat_with_image,
        inputs=[basic_input, basic_chatbot, basic_chat_state, image, basic_img_list, temperature],
        outputs=[basic_chatbot, basic_chat_state, basic_img_list]
    ).then(lambda: "", None, basic_input)
    
    enhanced_send.click(
        lambda *args: process_chat_with_image(*args, is_enhanced=True),
        inputs=[enhanced_input, enhanced_chatbot, enhanced_chat_state, image, enhanced_img_list, temperature],
        outputs=[enhanced_chatbot, enhanced_chat_state, enhanced_img_list]
    ).then(lambda: "", None, enhanced_input)
    
    enhanced_input.submit(
        lambda *args: process_chat_with_image(*args, is_enhanced=True),
        inputs=[enhanced_input, enhanced_chatbot, enhanced_chat_state, image, enhanced_img_list, temperature],
        outputs=[enhanced_chatbot, enhanced_chat_state, enhanced_img_list]
    ).then(lambda: "", None, enhanced_input)
    
    clear.click(
        reset_chat,
        inputs=[basic_chat_state, basic_img_list],
        outputs=[basic_chatbot, basic_chat_state, basic_img_list]
    ).then(
        reset_chat,
        inputs=[enhanced_chat_state, enhanced_img_list],
        outputs=[enhanced_chatbot, enhanced_chat_state, enhanced_img_list]
    ).then(lambda: None, None, image)
    
    # Knowledge Graph events
    demo.load(fn=create_knowledge_graph, inputs=None, outputs=graph_plot)
    reload_graph.click(fn=create_knowledge_graph, inputs=None, outputs=graph_plot)
    show_btn.click(draw_crop_from_csv, inputs=[crop_id], outputs=[graph_plot])

# Launch with custom settings
if __name__ == "__main__":
    demo.queue(max_size=20)
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
