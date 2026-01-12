#!/usr/bin/env python3
"""
Refusal Vector Extraction

Computes the refusal direction vector v_refuse as the mean difference
between refused and accepted responses at t_post-inst layer.

Usage:
    python extraction/refusal_vector.py --model MODEL --refused_file FILE --accepted_file FILE --output_dir outputs
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import formatInp_llama_persuasion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_model_layers(model: AutoModelForCausalLM) -> List[nn.Module]:
    """Get the transformer layers from different model architectures."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return list(model.model.layers)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return list(model.transformer.h)
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return list(model.gpt_neox.layers)
    else:
        raise ValueError("Unknown model architecture")


def extract_activations_at_post_inst(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    layer: int,
    model_type: str = 'llama2',
) -> Tensor:
    """
    Extract activations at the post-instruction position (last token of input).
    
    For t_post-inst, we extract at the last token position which corresponds
    to the end of the formatted input prompt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompt strings
        layer: Layer index to extract from
        model_type: Type of model for prompt formatting
        
    Returns:
        Tensor of activations [num_prompts, hidden_dim]
    """
    model.eval()
    block_modules = get_model_layers(model)
    
    activations = []
    activation_storage = [None]
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        # Extract at last token position (t_post-inst)
        activation_storage[0] = hidden_states[:, -1, :].detach().cpu()
    
    handle = block_modules[layer].register_forward_hook(hook_fn)
    
    try:
        for prompt in tqdm(prompts, desc=f"Extracting L{layer} activations"):
            # Format prompt with template
            formatted = formatInp_llama_persuasion({'instruction': prompt}, model=model_type)
            
            inputs = tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(model.device)
            
            with torch.no_grad():
                _ = model(**inputs)
            
            if activation_storage[0] is not None:
                activations.append(activation_storage[0])
    finally:
        handle.remove()
    
    return torch.cat(activations, dim=0)


def load_prompts_from_file(file_path: str, limit: Optional[int] = None) -> List[str]:
    """
    Load prompts from a JSON/JSONL file.
    
    Args:
        file_path: Path to the file
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
                prompt = data.get('prompt') or data.get('instruction') or data.get('question') or data.get('bad_q')
                if prompt:
                    prompts.append(prompt)
        else:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        prompt = item.get('prompt') or item.get('instruction') or item.get('question') or item.get('bad_q')
                    else:
                        prompt = item
                    if prompt:
                        prompts.append(prompt)
    
    if limit:
        prompts = prompts[:limit]
    
    logger.info(f"Loaded {len(prompts)} prompts from {file_path}")
    return prompts


def load_prompts_with_refusal_labels(
    results_file: str,
    limit: Optional[int] = None
) -> tuple[List[str], List[str]]:
    """
    Load prompts from inference results file, separated by refusal status.
    
    Args:
        results_file: Path to JSONL file with 'prompt' and 'refused' fields
        limit: Optional limit per category
        
    Returns:
        Tuple of (refused_prompts, accepted_prompts)
    """
    refused = []
    accepted = []
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            prompt = data.get('prompt', '')
            if data.get('refused', False):
                refused.append(prompt)
            else:
                accepted.append(prompt)
    
    if limit:
        refused = refused[:limit]
        accepted = accepted[:limit]
    
    logger.info(f"Loaded {len(refused)} refused + {len(accepted)} accepted from {results_file}")
    return refused, accepted


def main():
    parser = argparse.ArgumentParser(description="Compute refusal direction vector")
    parser.add_argument('--model', type=str, required=True,
                       help="Model path or HuggingFace model name")
    parser.add_argument('--model_type', type=str, default='llama2',
                       choices=['llama2', 'llama3', 'qwen', 'vicuna'])
    
    # Input options - either separate files or a single results file
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--results_file', type=str,
                            help="Path to inference results JSONL with refusal labels")
    input_group.add_argument('--refused_file', type=str,
                            help="Path to file with refused prompts")
    
    parser.add_argument('--accepted_file', type=str,
                       help="Path to file with accepted prompts (required if using --refused_file)")
    
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help="Output directory")
    parser.add_argument('--layer', type=int, default=None,
                       help="Layer for post-instruction encoding (default: num_layers - 1)")
    parser.add_argument('--limit', type=int, default=None,
                       help="Limit prompts per category")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.refused_file and not args.accepted_file:
        parser.error("--accepted_file is required when using --refused_file")
    
    # Load model
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
    
    # Determine layer (t_post-inst is typically the last layer)
    num_layers = model.config.num_hidden_layers
    layer = args.layer if args.layer is not None else num_layers - 1
    logger.info(f"Using layer {layer} (L_post for t_post-inst)")
    
    # Load prompts
    if args.results_file:
        refused_prompts, accepted_prompts = load_prompts_with_refusal_labels(
            args.results_file, args.limit
        )
    else:
        refused_prompts = load_prompts_from_file(args.refused_file, args.limit)
        accepted_prompts = load_prompts_from_file(args.accepted_file, args.limit)
    
    if not refused_prompts:
        logger.error("No refused prompts found!")
        return
    if not accepted_prompts:
        logger.error("No accepted prompts found!")
        return
    
    # Extract activations
    logger.info("Extracting activations for refused prompts...")
    refused_acts = extract_activations_at_post_inst(
        model, tokenizer, refused_prompts, layer, args.model_type
    )
    
    logger.info("Extracting activations for accepted prompts...")
    accepted_acts = extract_activations_at_post_inst(
        model, tokenizer, accepted_prompts, layer, args.model_type
    )
    
    # Compute refusal direction: v_refuse = mean_refused - mean_accepted
    mean_refused = refused_acts.mean(dim=0)
    mean_accepted = accepted_acts.mean(dim=0)
    
    v_refuse = mean_refused - mean_accepted
    v_refuse_normalized = v_refuse / (torch.norm(v_refuse) + 1e-8)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(v_refuse_normalized, output_dir / "v_refuse.pt")
    logger.info(f"Saved v_refuse.pt (shape: {v_refuse_normalized.shape})")
    
    # Also save raw means for analysis
    torch.save(mean_refused, output_dir / "mean_refused.pt")
    torch.save(mean_accepted, output_dir / "mean_accepted.pt")
    
    # Print statistics
    print("\n" + "="*60)
    print("Refusal Vector Statistics:")
    print("-"*60)
    print(f"Vector norm (before normalization): {torch.norm(v_refuse).item():.4f}")
    print(f"Refused activations: {refused_acts.shape[0]} samples")
    print(f"Accepted activations: {accepted_acts.shape[0]} samples")
    print(f"Hidden dimension: {v_refuse.shape[0]}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
