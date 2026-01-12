#!/usr/bin/env python3
"""
Run Inference Script

Generates model responses for input prompts and labels them with refusal status.
This script supports multiple model architectures (Llama2, Llama3, Qwen) and
produces JSONL outputs with prompt, generation, and refusal labels.

Usage:
    python scripts/run_inference.py --model MODEL_PATH --input INPUT_FILE --output OUTPUT_FILE
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm

# Add src to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import REFUSAL_PHRASE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_prompts(input_path: str) -> List[Dict[str, Any]]:
    """
    Load prompts from JSONL or plain text file.
    
    Args:
        input_path: Path to input file (JSONL or text with one prompt per line)
        
    Returns:
        List of dictionaries with 'prompt' key
    """
    prompts = []
    path = Path(input_path)
    
    # Try JSONL first
    if path.suffix in ['.jsonl', '.json']:
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix == '.jsonl':
                for line in f:
                    data = json.loads(line.strip())
                    # Handle different field names
                    if 'prompt' in data:
                        prompts.append(data)
                    elif 'instruction' in data:
                        prompts.append({'prompt': data['instruction'], **data})
                    elif 'question' in data:
                        prompts.append({'prompt': data['question'], **data})
                    elif 'bad_q' in data:
                        prompts.append({'prompt': data['bad_q'], **data})
            else:
                # Regular JSON
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            if 'prompt' in item:
                                prompts.append(item)
                            elif 'instruction' in item:
                                prompts.append({'prompt': item['instruction'], **item})
                            elif 'question' in item:
                                prompts.append({'prompt': item['question'], **item})
                            elif 'bad_q' in item:
                                prompts.append({'prompt': item['bad_q'], **item})
                        elif isinstance(item, str):
                            prompts.append({'prompt': item})
    else:
        # Plain text file
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append({'prompt': line})
    
    logger.info(f"Loaded {len(prompts)} prompts from {input_path}")
    return prompts


def get_prompt_template(model_type: str) -> str:
    """
    Get the appropriate prompt template for the model type.
    
    Args:
        model_type: Type of model ('llama2', 'llama3', 'qwen', 'vicuna')
        
    Returns:
        Prompt template string with {} placeholder
    """
    templates = {
        'llama2': "[INST]{}[/INST]",
        'llama3': "<|start_header_id|>user<|end_header_id|>\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
        'qwen': "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant",
        'vicuna': "USER:{}\nASSISTANT:",
    }
    return templates.get(model_type, "[INST]{}[/INST]")


def detect_refusal(response: str) -> bool:
    """
    Detect if a response is a refusal based on common refusal phrases.
    
    Args:
        response: Model-generated response text
        
    Returns:
        True if the response appears to be a refusal
    """
    response_lower = response.lower()
    
    for phrase in REFUSAL_PHRASE:
        if phrase.lower() in response_lower:
            return True
    
    return False


def extract_response(full_output: str, model_type: str, input_prompt: str = "") -> str:
    """
    Extract the model's response from the full generated text.
    
    Args:
        full_output: Complete generated text
        model_type: Type of model used
        input_prompt: Original input prompt
        
    Returns:
        Extracted model response
    """
    if model_type == 'llama2':
        if '[/INST]' in full_output:
            return full_output.split('[/INST]')[-1].strip()
    elif model_type == 'llama3':
        if 'assistant<|end_header_id|>' in full_output:
            return full_output.split('assistant<|end_header_id|>')[-1].strip()
        if input_prompt:
            return full_output[len(input_prompt):].strip()
    elif model_type == 'qwen':
        if '<|im_start|>assistant' in full_output:
            return full_output.split('<|im_start|>assistant')[-1].replace('<|im_end|>', '').strip()
    elif model_type == 'vicuna':
        if 'ASSISTANT:' in full_output:
            return full_output.split('ASSISTANT:')[-1].strip()
    
    # Fallback: return everything after the input
    if input_prompt and input_prompt in full_output:
        return full_output[len(input_prompt):].strip()
    
    return full_output.strip()


def run_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[Dict[str, Any]],
    model_type: str,
    max_new_tokens: int = 256,
    batch_size: int = 1
) -> List[Dict[str, Any]]:
    """
    Run inference on a list of prompts.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompt dictionaries
        model_type: Type of model
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for inference
        
    Returns:
        List of results with prompt, generation, and refusal status
    """
    results = []
    template = get_prompt_template(model_type)
    
    # Generation config
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    
    model.eval()
    
    for item in tqdm(prompts, desc="Running inference"):
        prompt_text = item['prompt']
        formatted_prompt = template.format(prompt_text)
        
        # Tokenize
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=gen_config,
            )
        
        # Decode
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = extract_response(full_output, model_type, formatted_prompt)
        
        # Detect refusal
        refused = detect_refusal(response)
        
        # Build result
        result = {
            'prompt': prompt_text,
            'generation': response,
            'refused': refused,
            'formatted_prompt': formatted_prompt,
        }
        
        # Copy any additional fields from input
        for key, value in item.items():
            if key not in result:
                result[key] = value
        
        results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference and label refusals")
    parser.add_argument('--model', type=str, required=True, 
                       help="Model path or HuggingFace model name")
    parser.add_argument('--input', type=str, required=True,
                       help="Input file path (JSONL or text)")
    parser.add_argument('--output', type=str, required=True,
                       help="Output JSONL file path")
    parser.add_argument('--model_type', type=str, default='llama2',
                       choices=['llama2', 'llama3', 'qwen', 'vicuna'],
                       help="Type of model for prompt formatting")
    parser.add_argument('--max_new_tokens', type=int, default=256,
                       help="Maximum new tokens to generate")
    parser.add_argument('--batch_size', type=int, default=1,
                       help="Batch size for inference")
    parser.add_argument('--device', type=str, default='auto',
                       help="Device to use ('auto', 'cuda', 'cpu')")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model}")
    
    device_map = "auto" if args.device == "auto" else args.device
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )
    
    # Load prompts
    prompts = load_prompts(args.input)
    
    # Run inference
    results = run_inference(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        model_type=args.model_type,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # Print summary
    refused_count = sum(1 for r in results if r['refused'])
    accepted_count = len(results) - refused_count
    
    logger.info(f"Results saved to {args.output}")
    logger.info(f"Total: {len(results)}, Refused: {refused_count}, Accepted: {accepted_count}")


if __name__ == "__main__":
    main()
