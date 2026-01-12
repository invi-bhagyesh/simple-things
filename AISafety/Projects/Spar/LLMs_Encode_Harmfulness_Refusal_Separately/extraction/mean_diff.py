#!/usr/bin/env python3
"""
Mean Difference Vector Computation

Computes baseline harmfulness and refusal mean difference vectors.
Extracts hidden states from the model and computes mean differences
between harmful/harmless and refused/accepted examples.

Usage:
    python extraction/mean_diff.py --model MODEL_PATH --data_dir DATA_DIR --output_dir OUTPUT_DIR
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any, Tuple, Callable, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import formatInp_llama_persuasion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ActivationCache:
    """Cache for storing activations during forward pass."""
    
    def __init__(self):
        self.activations = {}
    
    def clear(self):
        self.activations = {}
    
    def store(self, layer: int, position: str, activation: Tensor):
        key = f"{layer}_{position}"
        self.activations[key] = activation
    
    def get(self, layer: int, position: str) -> Optional[Tensor]:
        key = f"{layer}_{position}"
        return self.activations.get(key)


def get_activation_hook(cache: ActivationCache, layer: int, position_key: str, token_idx: int):
    """
    Creates a hook function to capture activations at a specific layer and token.
    
    Args:
        cache: ActivationCache to store activations
        layer: Layer index
        position_key: Key for the position (e.g., 't_inst', 't_post_inst')
        token_idx: Token index to extract (-1 for last token, -2 for second-to-last, etc.)
    
    Returns:
        Hook function
    """
    def hook_fn(module, input, output):
        # Output is typically a tuple (hidden_states, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # Extract activation at specific token position
        # hidden_states shape: [batch, seq_len, hidden_dim]
        activation = hidden_states[:, token_idx, :].detach().cpu()
        cache.store(layer, position_key, activation)
    
    return hook_fn


def get_model_layers(model: AutoModelForCausalLM) -> List[nn.Module]:
    """
    Get the transformer layers from different model architectures.
    
    Args:
        model: The language model
        
    Returns:
        List of transformer layer modules
    """
    # Try different architectures
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return list(model.model.layers)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return list(model.transformer.h)
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return list(model.gpt_neox.layers)
    else:
        raise ValueError("Unknown model architecture - cannot find transformer layers")


def extract_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    layers: List[int],
    model_type: str = 'llama2',
) -> Dict[int, Tensor]:
    """
    Extract activations from specified layers for all prompts.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompt strings
        layers: List of layer indices to extract from
        model_type: Type of model for prompt formatting
        
    Returns:
        Dictionary mapping layer index to stacked activations [num_prompts, hidden_dim]
    """
    model.eval()
    block_modules = get_model_layers(model)
    
    # Storage for activations per layer
    layer_activations = {layer: [] for layer in layers}
    
    cache = ActivationCache()
    handles = []
    
    # Register hooks for all specified layers
    for layer in layers:
        hook = get_activation_hook(cache, layer, 'last', -1)
        handles.append(block_modules[layer].register_forward_hook(hook))
    
    try:
        for prompt in tqdm(prompts, desc="Extracting activations"):
            cache.clear()
            
            # Format and tokenize
            formatted = formatInp_llama_persuasion({'instruction': prompt}, model=model_type)
            inputs = tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(model.device)
            
            # Forward pass
            with torch.no_grad():
                _ = model(**inputs)
            
            # Collect activations
            for layer in layers:
                act = cache.get(layer, 'last')
                if act is not None:
                    layer_activations[layer].append(act)
    finally:
        # Remove hooks
        for handle in handles:
            handle.remove()
    
    # Stack activations for each layer
    result = {}
    for layer, acts in layer_activations.items():
        if acts:
            result[layer] = torch.cat(acts, dim=0)  # [num_prompts, hidden_dim]
    
    return result


def compute_mean_difference(
    acts_group1: Tensor,
    acts_group2: Tensor,
    normalize: bool = True
) -> Tensor:
    """
    Compute mean difference vector between two groups of activations.
    
    Args:
        acts_group1: Activations from first group [N1, hidden_dim]
        acts_group2: Activations from second group [N2, hidden_dim]
        normalize: Whether to normalize the resulting vector
        
    Returns:
        Mean difference vector [hidden_dim]
    """
    mean1 = acts_group1.mean(dim=0)
    mean2 = acts_group2.mean(dim=0)
    diff = mean1 - mean2
    
    if normalize:
        diff = diff / (torch.norm(diff) + 1e-8)
    
    return diff


def load_dataset_prompts(file_path: str, limit: Optional[int] = None) -> List[str]:
    """
    Load prompts from a dataset file.
    
    Args:
        file_path: Path to JSON/JSONL file
        limit: Optional limit on number of prompts
        
    Returns:
        List of prompt strings
    """
    prompts = []
    path = Path(file_path)
    
    with open(path, 'r', encoding='utf-8') as f:
        if path.suffix == '.jsonl':
            for line in f:
                data = json.loads(line.strip())
                if 'prompt' in data:
                    prompts.append(data['prompt'])
                elif 'instruction' in data:
                    prompts.append(data['instruction'])
                elif 'question' in data:
                    prompts.append(data['question'])
                elif 'bad_q' in data:
                    prompts.append(data['bad_q'])
        else:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        if 'prompt' in item:
                            prompts.append(item['prompt'])
                        elif 'instruction' in item:
                            prompts.append(item['instruction'])
                        elif 'question' in item:
                            prompts.append(item['question'])
                        elif 'bad_q' in item:
                            prompts.append(item['bad_q'])
                    elif isinstance(item, str):
                        prompts.append(item)
    
    if limit:
        prompts = prompts[:limit]
    
    logger.info(f"Loaded {len(prompts)} prompts from {file_path}")
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Compute mean difference vectors")
    parser.add_argument('--model', type=str, required=True,
                       help="Model path or HuggingFace model name")
    parser.add_argument('--harmful_file', type=str, required=True,
                       help="Path to harmful prompts file")
    parser.add_argument('--harmless_file', type=str, required=True,
                       help="Path to harmless prompts file")
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help="Output directory")
    parser.add_argument('--model_type', type=str, default='llama2',
                       choices=['llama2', 'llama3', 'qwen', 'vicuna'],
                       help="Type of model")
    parser.add_argument('--layer_inst', type=int, default=None,
                       help="Layer for instruction encoding (default: num_layers // 2)")
    parser.add_argument('--layer_post', type=int, default=None,
                       help="Layer for post-instruction encoding (default: num_layers - 1)")
    parser.add_argument('--limit', type=int, default=None,
                       help="Limit number of prompts per category")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Determine layers
    num_layers = model.config.num_hidden_layers
    layer_inst = args.layer_inst if args.layer_inst is not None else num_layers // 2
    layer_post = args.layer_post if args.layer_post is not None else num_layers - 1
    layers = [layer_inst, layer_post]
    
    logger.info(f"Using layers: L_inst={layer_inst}, L_post={layer_post}")
    
    # Load prompts
    harmful_prompts = load_dataset_prompts(args.harmful_file, args.limit)
    harmless_prompts = load_dataset_prompts(args.harmless_file, args.limit)
    
    # Extract activations
    logger.info("Extracting activations for harmful prompts...")
    harmful_acts = extract_activations(model, tokenizer, harmful_prompts, layers, args.model_type)
    
    logger.info("Extracting activations for harmless prompts...")
    harmless_acts = extract_activations(model, tokenizer, harmless_prompts, layers, args.model_type)
    
    # Compute mean difference vectors
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mean harm direction (harmful - harmless) at instruction layer
    mean_harm = compute_mean_difference(
        harmful_acts[layer_inst],
        harmless_acts[layer_inst],
        normalize=True
    )
    torch.save(mean_harm, output_dir / "mean_harm.pt")
    logger.info(f"Saved mean_harm.pt (shape: {mean_harm.shape})")
    
    # Also save the raw means
    torch.save(harmful_acts[layer_inst].mean(dim=0), output_dir / "mean_harmful_acts.pt")
    torch.save(harmless_acts[layer_inst].mean(dim=0), output_dir / "mean_harmless_acts.pt")
    
    logger.info(f"Mean difference vectors saved to {output_dir}")
    logger.info("Note: For refused/accepted vectors, use lat_probe.py with labeled data")


if __name__ == "__main__":
    main()
