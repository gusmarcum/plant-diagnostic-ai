"""
 SPDX-License-Identifier: BSD-3-Clause
"""

import os
import sys
from omegaconf import OmegaConf

from minigpt4.common.registry import registry
from .builders.base_dataset_builder import BaseDatasetBuilder
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.tasks import *

# Initialize paths
root_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.join(root_dir, "..")

# Expose a general build_datasets function
def build_datasets(config_path):
    # Load configuration
    config = OmegaConf.load(config_path)

    # Iterate over datasets in config and build each one
    datasets = {}
    for dataset_name, dataset_config in config.datasets.items():
        builder_cls = registry.get_builder_class(dataset_name)
        if builder_cls is None:
            raise ValueError(f"No builder registered for dataset {dataset_name}")

        # Initialize the builder with the dataset-specific configuration
        builder = builder_cls(dataset_config)
        datasets[dataset_name] = builder.build_datasets()

    return datasets

