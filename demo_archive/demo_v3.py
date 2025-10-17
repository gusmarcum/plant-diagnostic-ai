import argparse
import os
import random
from collections import defaultdict
import cv2
import re
import numpy as np
from PIL import Image, Image as PILImage
import torch
import html
import gradio as gr
from serpapi import GoogleSearch

import torchvision.transforms as T
import torch.backends.cudnn as cudnn

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from functools import lru_cache
import logging

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# Import modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

# Configure logging
logging.basicConfig(filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.ERROR)

# Define conversation template
CONV_VISION = Conversation(
    system="",
    roles=(r"<s>[INST] ", r" [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

# SERPAPI Configuration
SERP_API_KEY = 'fb6c393ddf8c395e92e9f5e4244f1e3ee75084f85c94aabde8954cd12c5df1ea'
if not SERP_API_KEY:
    raise ValueError("SERP_API_KEY environment variable not set.")

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Demo_v3")
    parser.add_argument("--cfg-path", default="eval_configs/minigptv2_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    return parser.parse_args()

args = parse_args()

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

# Initialize visual processor
vis_processor_cfg = cfg.datasets_cfg.coco_vqa.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

# Initialize chat
chat = Chat(model, vis_processor, device=device)
print('Initialization Finished')

def fetch_serp_context(query):
    """Fetch additional context from SERPAPI with improved query."""
    # Make the search query more specific
    enhanced_query = f"agricultural robot technology {query} technical specifications capabilities"
    
    params = {
        "engine": "google",
        "q": enhanced_query,
        "api_key": SERP_API_KEY,
        "num": 5,
        "gl": "us",
        "hl": "en",
    }
    
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results", [])
        logging.info(f"Found {len(organic_results)} SERP results")
        
        # Better context extraction with detailed logging
        contexts = []
        for idx, result in enumerate(organic_results[:5]):
            title = result.get("title", "").strip()
            snippet = result.get("snippet", "").strip()
            
            # Skip truncated or empty content
            if not snippet or snippet.endswith('...'):
                continue
                
            # Clean up common noise
            snippet = snippet.replace('Comments5', '')
            snippet = snippet.replace('AgDay', '')
            snippet = ' '.join(snippet.split())  # Normalize whitespace
            
            contexts.append(snippet)
            logging.debug(f"Result {idx + 1}:\nTitle: {title}\nCleaned snippet: {snippet}\n")
        
        context = " ".join(contexts)
        logging.info(f"Final cleaned context length: {len(context)} characters")
        logging.debug(f"Final cleaned context: {context}")
        return context
        
    except Exception as e:
        logging.error(f"SERPAPI Error: {str(e)}")
        return ""

def create_knowledge_graph():
    """Create knowledge graph visualization from FAOSTAT data."""
    try:
        nodes_path = 'knowledge_graph/kg_nodes_faostat.csv'
        relationships_path = 'knowledge_graph/kg_relationships_faostat.csv'

        if not os.path.exists(nodes_path):
            raise FileNotFoundError(f"Nodes file not found: {nodes_path}")
        if not os.path.exists(relationships_path):
            raise FileNotFoundError(f"Relationships file not found: {relationships_path}")

        print("Reading CSV files...")
        nodes_df = pd.read_csv(nodes_path)
        relationships_df = pd.read_csv(relationships_path)

        print(f"Loaded {len(nodes_df)} nodes and {len(relationships_df)} relationships")

        G = nx.DiGraph()

        # Add nodes and edges
        for _, row in nodes_df.iterrows():
            G.add_node(row['id'], label=row['label'], name=row['name'])

        for _, row in relationships_df.head(1000).iterrows():
            G.add_edge(row['start_id'], row['end_id'], type=row['type'])

        # Create visualization
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Create edge trace
        edge_x, edge_y, edge_text = [], [], []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(edge[2]['type'])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines'
        )

        # Create node trace
        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{G.nodes[node]['label']}: {G.nodes[node]['name']}")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                color=[G.degree(node) for node in G.nodes()],
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                )
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='FAOSTAT Knowledge Graph',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           width=800,
                           height=600
                       ))

        return fig
    except Exception as e:
        logging.error(f"Error creating knowledge graph: {str(e)}")
        return go.Figure()

def handle_image_upload(gr_img, chat_state, img_list, upload_flag, replace_flag):
    """Handle image upload and update chat state."""
    print("=== Image Upload Handler ===")
    print(f"Input image type: {type(gr_img)}")
    
    if gr_img is None:
        print("No image provided")
        return chat_state, img_list, upload_flag, replace_flag

    # Process image
    if isinstance(gr_img, dict):
        print("Image is a dict, extracting image")
        image = gr_img['image']
    else:
        image = gr_img

    print(f"Processed image type: {type(image)}")
    
    if isinstance(image, PILImage.Image):
        print("Processing PIL Image")
        # Convert PIL Image to tensor and add batch dimension
        image_tensor = vis_processor(image).unsqueeze(0).to(device)
        print(f"Image tensor shape: {image_tensor.shape}")
        
        # Encode image using model's encoder
        with torch.no_grad():
            image_emb, _ = model.encode_img(image_tensor)
        print(f"Encoded image shape: {image_emb.shape}")
        
        img_list = []
        chat_state = CONV_VISION.copy()
        print(f"Chat state messages before upload: {chat_state.messages}")
        # Initialize chat with image
        chat.upload_img(image_emb, chat_state, img_list)
        print(f"Chat state messages after upload: {chat_state.messages}")
        print(f"Image list length: {len(img_list)}")
        if img_list:
            print(f"First image in list shape: {img_list[0].shape}")
    else:
        print(f"Image is not a PIL Image: {type(image)}")

    return chat_state, img_list, upload_flag, replace_flag

def validate_enhanced_response(basic_response, enhanced_response):
    """Check if enhanced response is significantly different from basic response."""
    if len(enhanced_response) < len(basic_response) * 2:
        logging.warning("Enhanced response not significantly longer than basic response")
        return False
    return True

def clean_response(response):
    """Clean up coordinate numbers and other artifacts from the response."""
    # Remove coordinate patterns like {<2><12><64><80>}
    cleaned = re.sub(r'\{<\d+><\d+><\d+><\d+>\}', '', response)
    # Remove extra whitespace while preserving sentence structure
    cleaned = ' '.join(cleaned.split())
    return cleaned.strip()

def process_chat_with_image(user_message, chatbot, chat_state, gr_img, img_list, upload_flag, replace_flag, temperature=0.7, is_enhanced=False):
    """Process chat with improved enhanced response generation."""
    try:
        logging.info(f"\n=== Process Chat With Image ===")
        logging.info(f"Message: {user_message}")
        logging.info(f"Temperature: {temperature}")
        logging.info(f"Enhanced mode: {is_enhanced}")
        
        # Get basic response
        chat.ask(user_message, chat_state)
        basic_response = chat.answer(conv=chat_state,
                                   img_list=img_list,
                                   temperature=temperature,
                                   max_new_tokens=500,
                                   max_length=2000)[0]
        basic_response = clean_response(basic_response)
        
        if not is_enhanced:
            return chatbot + [[user_message, basic_response]], chat_state, img_list, upload_flag, replace_flag

        # Enhanced response generation
        logging.info(f"Base response: {basic_response}")
        serp_context = fetch_serp_context(basic_response)
        
        if serp_context:
            enhanced_prompt = (
                f"Looking at this agricultural robot image, provide a detailed technical analysis. "
                f"Additional context about similar robots: {serp_context}\n\n"
                f"Describe:\n"
                f"- What you see in the image (components, design, setting)\n"
                f"- The robot's likely capabilities based on its visible features\n"
                f"- Its probable agricultural applications\n"
                f"- Any relevant technical specifications from the context\n\n"
                f"Respond in a natural, flowing way that combines your visual analysis with the technical context."
            )
            
            logging.info(f"Enhanced prompt: {enhanced_prompt}")
            chat.ask(enhanced_prompt, chat_state)
            
            enhanced_response = chat.answer(
                conv=chat_state,
                img_list=img_list,
                temperature=temperature,
                max_new_tokens=1000,
                max_length=2000
            )[0]
            
            # Validate response quality
            if len(enhanced_response) < 100:  # If response is too short
                retry_prompt = (
                    f"Please provide a more detailed analysis of the robot in the image. "
                    f"Describe what you see and incorporate relevant technical details about its capabilities and applications."
                )
                chat.ask(retry_prompt, chat_state)
                enhanced_response = chat.answer(
                    conv=chat_state,
                    img_list=img_list,
                    temperature=temperature,
                    max_new_tokens=1000,
                    max_length=2000
                )[0]
            
            logging.info(f"Enhanced response: {enhanced_response}")
            return chatbot + [[user_message, enhanced_response]], chat_state, img_list, upload_flag, replace_flag

        return chatbot + [[user_message, basic_response]], chat_state, img_list, upload_flag, replace_flag

    except Exception as e:
        logging.error(f"Error in chat processing: {str(e)}")
        return chatbot + [[user_message, f"Error: {str(e)}"]], chat_state, img_list, upload_flag, replace_flag

def reset_chat(chat_state, img_list, upload_flag, replace_flag):
    """Reset chat state and image list."""
    return [], None, "", CONV_VISION.copy(), [], 0, 0

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# MiniGPT-v2 Demo with Knowledge Graph")
    
    # Knowledge Graph Section
    graph_plot = gr.Plot(label="Knowledge Graph", value=create_knowledge_graph())
    
    with gr.Row():
        # Image Upload Section
        with gr.Column(scale=1):
            image = gr.Image(type="pil")
            temperature = gr.Slider(
                minimum=0.1,
                maximum=1.5,
                value=0.7,
                step=0.1,
                label="Temperature",
                interactive=True
            )
            clear = gr.Button("Reset")
        
        # Chat Section
        with gr.Column(scale=2):
            # Basic Chat
            gr.Markdown("### Basic Chat")
            basic_chatbot = gr.Chatbot()
            basic_chat_state = gr.State(CONV_VISION.copy())
            basic_img_list = gr.State([])
            basic_upload_flag = gr.State(value=0)
            basic_replace_flag = gr.State(value=0)
            
            with gr.Row():
                basic_input = gr.Textbox(placeholder="Enter message for basic chat...", scale=8)
                basic_send = gr.Button("Send", scale=1)
            
            # Enhanced Chat
            gr.Markdown("### Enhanced Chat (with SERPAPI)")
            enhanced_chatbot = gr.Chatbot()
            enhanced_chat_state = gr.State(CONV_VISION.copy())
            enhanced_img_list = gr.State([])
            enhanced_upload_flag = gr.State(value=0)
            enhanced_replace_flag = gr.State(value=0)
            
            with gr.Row():
                enhanced_input = gr.Textbox(placeholder="Enter message for enhanced chat...", scale=8)
                enhanced_send = gr.Button("Send", scale=1)
    
    # Event handlers
    # Image upload handler
    image.change(
        handle_image_upload,
        inputs=[image, basic_chat_state, basic_img_list, basic_upload_flag, basic_replace_flag],
        outputs=[basic_chat_state, basic_img_list, basic_upload_flag, basic_replace_flag]
    ).then(
        handle_image_upload,
        inputs=[image, enhanced_chat_state, enhanced_img_list, enhanced_upload_flag, enhanced_replace_flag],
        outputs=[enhanced_chat_state, enhanced_img_list, enhanced_upload_flag, enhanced_replace_flag]
    )
    
    # Basic chat handler
    basic_send.click(
        process_chat_with_image,
        inputs=[
            basic_input, basic_chatbot, basic_chat_state, image, basic_img_list,
            basic_upload_flag, basic_replace_flag, temperature
        ],
        outputs=[basic_chatbot, basic_chat_state, basic_img_list, basic_upload_flag, basic_replace_flag]
    )
    
    # Enhanced chat handler
    enhanced_send.click(
        lambda *args: process_chat_with_image(*args, is_enhanced=True),
        inputs=[
            enhanced_input, enhanced_chatbot, enhanced_chat_state, image, enhanced_img_list,
            enhanced_upload_flag, enhanced_replace_flag, temperature
        ],
        outputs=[enhanced_chatbot, enhanced_chat_state, enhanced_img_list, enhanced_upload_flag, enhanced_replace_flag]
    )
    
    # Reset handler
    clear.click(
        reset_chat,
        inputs=[basic_chat_state, basic_img_list, basic_upload_flag, basic_replace_flag],
        outputs=[basic_chatbot, image, basic_input, basic_chat_state, basic_img_list, basic_upload_flag, basic_replace_flag]
    ).then(
        reset_chat,
        inputs=[enhanced_chat_state, enhanced_img_list, enhanced_upload_flag, enhanced_replace_flag],
        outputs=[enhanced_chatbot, image, enhanced_input, enhanced_chat_state, enhanced_img_list, enhanced_upload_flag, enhanced_replace_flag]
    )

demo.queue()
demo.launch(share=True)

