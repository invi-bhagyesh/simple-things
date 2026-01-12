#!/usr/bin/env python3
"""
LAT (Latent Adversarial Training) Probe

Trains a linear readout for harmfulness at t_inst position.
Produces v_harm vector and per-example harmfulness scores.

Supports two modes:
- --fit: Train the LAT probe and save v_harm.pt
- --score: Score examples using a trained probe

Usage:
    python extraction/lat_probe.py --fit --model MODEL_PATH --harmful_file FILE --harmless_file FILE --output_dir outputs
    python extraction/lat_probe.py --score --model MODEL_PATH --input_file FILE --probe_path outputs/v_harm.pt --output outputs/scores.jsonl
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve

# Add src and extraction to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
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


def extract_activation_at_layer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    layer: int,
    model_type: str = 'llama2',
    token_position: int = -1,  # -1 for last token of instruction
) -> Tensor:
    """
    Extract activations at a specific layer and token position for all prompts.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompt strings
        layer: Layer index to extract from
        model_type: Type of model for prompt formatting
        token_position: Token position (-1 for last, -2 for second-to-last, etc.)
        
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
        activation_storage[0] = hidden_states[:, token_position, :].detach().cpu()
    
    handle = block_modules[layer].register_forward_hook(hook_fn)
    
    try:
        for prompt in tqdm(prompts, desc=f"Extracting L{layer} activations"):
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


def fit_ridge_regression(H: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """
    Fit ridge regression for LAT.
    
    Args:
        H: Feature matrix [M, D]
        y: Labels [M]
        lam: Regularization parameter
        
    Returns:
        Normalized weight vector [D]
    """
    D = H.shape[1]
    
    # Ridge regression: w = (H^T H + λI)^{-1} H^T y
    HtH = H.T @ H
    HtY = H.T @ y
    
    # Solve the linear system
    w = np.linalg.solve(HtH + lam * np.eye(D), HtY)
    
    # Normalize
    w = w / (np.linalg.norm(w) + 1e-8)
    
    return w


def compute_scores(H: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute harmfulness scores.
    
    Args:
        H: Feature matrix [M, D]
        w: Weight vector [D]
        
    Returns:
        Scores [M]
    """
    return H @ w


def load_prompts_with_labels(
    harmful_file: str,
    harmless_file: str,
    limit: Optional[int] = None
) -> Tuple[List[str], np.ndarray]:
    """
    Load prompts from harmful and harmless files with corresponding labels.
    
    Args:
        harmful_file: Path to harmful prompts
        harmless_file: Path to harmless prompts
        limit: Optional limit per category
        
    Returns:
        Tuple of (prompts, labels) where labels are 1 for harmful, 0 for harmless
    """
    prompts = []
    labels = []
    
    # Load harmful prompts
    harmful_prompts = []
    with open(harmful_file, 'r', encoding='utf-8') as f:
        data = json.load(f) if harmful_file.endswith('.json') else [json.loads(l) for l in f]
        for item in data:
            if isinstance(item, dict):
                prompt = item.get('instruction') or item.get('prompt') or item.get('question') or item.get('bad_q')
            else:
                prompt = item
            if prompt:
                harmful_prompts.append(prompt)
    
    if limit:
        harmful_prompts = harmful_prompts[:limit]
    
    # Load harmless prompts
    harmless_prompts = []
    with open(harmless_file, 'r', encoding='utf-8') as f:
        data = json.load(f) if harmless_file.endswith('.json') else [json.loads(l) for l in f]
        for item in data:
            if isinstance(item, dict):
                prompt = item.get('instruction') or item.get('prompt') or item.get('question')
            else:
                prompt = item
            if prompt:
                harmless_prompts.append(prompt)
    
    if limit:
        harmless_prompts = harmless_prompts[:limit]
    
    # Combine with labels
    prompts = harmful_prompts + harmless_prompts
    labels = np.array([1.0] * len(harmful_prompts) + [0.0] * len(harmless_prompts))
    
    logger.info(f"Loaded {len(harmful_prompts)} harmful + {len(harmless_prompts)} harmless prompts")
    
    return prompts, labels


def fit_lat_probe(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    harmful_file: str,
    harmless_file: str,
    layer: int,
    model_type: str,
    output_dir: str,
    lam: float = 1e-3,
    limit: Optional[int] = None,
):
    """
    Fit the LAT probe and save v_harm.pt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        harmful_file: Path to harmful prompts
        harmless_file: Path to harmless prompts
        layer: Layer index for extraction
        model_type: Type of model
        output_dir: Output directory
        lam: Ridge regularization parameter
        limit: Optional limit on prompts per category
    """
    # Load prompts with labels
    prompts, labels = load_prompts_with_labels(harmful_file, harmless_file, limit)
    
    # Extract activations
    logger.info(f"Extracting activations at layer {layer}...")
    H = extract_activation_at_layer(model, tokenizer, prompts, layer, model_type)
    H_np = H.numpy().astype(np.float32)
    
    # Fit ridge regression
    logger.info(f"Fitting ridge regression (λ={lam})...")
    w = fit_ridge_regression(H_np, labels, lam)
    
    # Compute scores for evaluation
    scores = compute_scores(H_np, w)
    
    # Compute AUC
    auc = roc_auc_score(labels, scores)
    logger.info(f"Training AUC: {auc:.4f}")
    
    # Save v_harm
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    v_harm = torch.tensor(w, dtype=torch.float32)
    torch.save(v_harm, output_path / "v_harm.pt")
    logger.info(f"Saved v_harm.pt (shape: {v_harm.shape})")
    
    # Save scores
    scores_data = []
    for i, prompt in enumerate(prompts):
        scores_data.append({
            'prompt': prompt,
            'score': float(scores[i]),
            'label': 'harmful' if labels[i] == 1 else 'harmless',
            'label_binary': int(labels[i])
        })
    
    with open(output_path / "lat_scores.jsonl", 'w', encoding='utf-8') as f:
        for item in scores_data:
            f.write(json.dumps(item) + '\n')
    
    logger.info(f"Saved lat_scores.jsonl")
    
    # Print score statistics
    harmful_scores = scores[labels == 1]
    harmless_scores = scores[labels == 0]
    
    print("\n" + "="*60)
    print("Score Statistics:")
    print("-"*60)
    print(f"Harmful prompts:  mean={harmful_scores.mean():.4f}, std={harmful_scores.std():.4f}")
    print(f"Harmless prompts: mean={harmless_scores.mean():.4f}, std={harmless_scores.std():.4f}")
    print(f"AUC: {auc:.4f}")
    print("="*60 + "\n")


def score_prompts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_file: str,
    probe_path: str,
    layer: int,
    model_type: str,
    output_file: str,
):
    """
    Score prompts using a trained LAT probe.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        input_file: Path to input prompts (JSONL)
        probe_path: Path to v_harm.pt
        layer: Layer index for extraction
        model_type: Type of model
        output_file: Output JSONL file
    """
    # Load probe
    v_harm = torch.load(probe_path)
    v_harm_np = v_harm.numpy()
    
    # Load prompts
    prompts = []
    original_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            prompt = item.get('prompt') or item.get('instruction') or item.get('question')
            if prompt:
                prompts.append(prompt)
                original_data.append(item)
    
    logger.info(f"Loaded {len(prompts)} prompts from {input_file}")
    
    # Extract activations
    H = extract_activation_at_layer(model, tokenizer, prompts, layer, model_type)
    H_np = H.numpy().astype(np.float32)
    
    # Compute scores
    scores = compute_scores(H_np, v_harm_np)
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, item in enumerate(original_data):
            result = {**item, 'harmfulness_score': float(scores[i])}
            f.write(json.dumps(result) + '\n')
    
    logger.info(f"Saved scores to {output_file}")
    print(f"\nScore range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"Score mean: {scores.mean():.4f}, std: {scores.std():.4f}")


def main():
    parser = argparse.ArgumentParser(description="LAT Probe for harmfulness detection")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--fit', action='store_true', help="Train LAT probe")
    mode_group.add_argument('--score', action='store_true', help="Score examples using trained probe")
    
    # Common arguments
    parser.add_argument('--model', type=str, required=True,
                       help="Model path or HuggingFace model name")
    parser.add_argument('--model_type', type=str, default='llama2',
                       choices=['llama2', 'llama3', 'qwen', 'vicuna'])
    parser.add_argument('--layer', type=int, default=None,
                       help="Layer index (default: num_layers // 2)")
    
    # Fit mode arguments
    parser.add_argument('--harmful_file', type=str,
                       help="Path to harmful prompts (for --fit)")
    parser.add_argument('--harmless_file', type=str,
                       help="Path to harmless prompts (for --fit)")
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help="Output directory (for --fit)")
    parser.add_argument('--lambda_reg', type=float, default=1e-3,
                       help="Ridge regularization parameter")
    parser.add_argument('--limit', type=int, default=None,
                       help="Limit prompts per category")
    
    # Score mode arguments
    parser.add_argument('--input_file', type=str,
                       help="Path to input prompts JSONL (for --score)")
    parser.add_argument('--probe_path', type=str,
                       help="Path to v_harm.pt (for --score)")
    parser.add_argument('--output', type=str,
                       help="Output JSONL file (for --score)")
    
    args = parser.parse_args()
    
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
    
    # Determine layer
    num_layers = model.config.num_hidden_layers
    layer = args.layer if args.layer is not None else num_layers // 2
    logger.info(f"Using layer {layer} (L_lat)")
    
    if args.fit:
        if not args.harmful_file or not args.harmless_file:
            parser.error("--fit requires --harmful_file and --harmless_file")
        
        fit_lat_probe(
            model=model,
            tokenizer=tokenizer,
            harmful_file=args.harmful_file,
            harmless_file=args.harmless_file,
            layer=layer,
            model_type=args.model_type,
            output_dir=args.output_dir,
            lam=args.lambda_reg,
            limit=args.limit,
        )
    
    elif args.score:
        if not args.input_file or not args.probe_path:
            parser.error("--score requires --input_file and --probe_path")
        
        output = args.output or args.input_file.replace('.jsonl', '_scored.jsonl')
        
        score_prompts(
            model=model,
            tokenizer=tokenizer,
            input_file=args.input_file,
            probe_path=args.probe_path,
            layer=layer,
            model_type=args.model_type,
            output_file=output,
        )


if __name__ == "__main__":
    main()
