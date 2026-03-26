"""
Inference Script for Trained Models

Usage:
    python scripts/inference.py --checkpoint models/current/best_model.pt --prompt "Once upon a time"
"""

import torch
import torch.nn as nn
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.tokenization import TokenizerManager
from src.inference.loading import load_checkpoint_model


def load_model_and_tokenizer(checkpoint_path: str, tokenizer_path: str, device: str):
    """Load model and tokenizer from disk."""
    model, _, _ = load_checkpoint_model(checkpoint_path, device=device)

    # Load tokenizer
    tokenizer = TokenizerManager.from_pretrained(tokenizer_path)
    
    return model, tokenizer


def generate_text(
    model: nn.Module,
    tokenizer: TokenizerManager,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = 'cpu'
):
    """
    Generate text from a prompt.
    
    Args:
        model: Trained GPT model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling
        device: Device to run on
        
    Returns:
        Generated text
    """
    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0].tolist())
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate text with trained model")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='models/current/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='models/current/tokenizer.json',
        help='Path to tokenizer'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='Once upon a time',
        help='Input prompt'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=100,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Top-k sampling'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.95,
        help='Nucleus sampling'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device (cpu, cuda, mps)'
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model from: {args.checkpoint}")
    print(f"Loading tokenizer from: {args.tokenizer}")
    
    model, tokenizer = load_model_and_tokenizer(
        args.checkpoint,
        args.tokenizer,
        device
    )
    
    print(f"\nPrompt: {args.prompt}")
    print("-" * 70)
    
    # Generate
    generated_text = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device
    )
    
    print(generated_text)
    print("-" * 70)


if __name__ == "__main__":
    main()
