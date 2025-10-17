"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import sys
from omegaconf import OmegaConf

from minigpt4.common.registry import registry
from .builders.image_text_pair_builder import build_datasets  # Explicitly import the desired build_datasets
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.tasks import *

# Initialize paths
root_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.join(root_dir, "..")

# Load configuration if not already loaded
default_cfg_path = os.path.join(repo_root, "/data/AGAI/MiniGPT-4/eval_configs/minigptv2_eval.yaml")  # Adjust path if necessary
default_cfg = OmegaConf.load(default_cfg_path)

# Debugging output
print("Loaded Configuration:", default_cfg)

# Provide a fallback for the entire env key
default_cfg.env = default_cfg.get("env", {"cache_root": "/data/AGAI/MiniGPT-4/cache"})
cache_root = os.path.join(repo_root, default_cfg.env["cache_root"])

# Register paths in the registry
registry.register_path("library_root", root_dir)
registry.register_path("repo_root", repo_root)
registry.register_path("cache_root", cache_root)

# Other global registrations
registry.register("MAX_INT", sys.maxsize)
registry.register("SPLIT_NAMES", ["train", "val", "test"])

