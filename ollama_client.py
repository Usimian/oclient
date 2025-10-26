#!/usr/bin/env python3
"""
Ollama Client – Generate Text from a Prompt

Usage:
    # Simple usage (uses default model "gpt-oss:20b" with full GPU)
    python ollama_client.py "Tell me a short joke"

    # Show verbose output with timing and token statistics
    python ollama_client.py -v "Tell me a short joke"

    # Specify a different model
    python ollama_client.py -m llama3 "What is the capital of France?"

    # Disable streaming (wait for full response)
    python ollama_client.py --no-stream "Describe the ocean in one sentence"
    
    # Use CPU only (no GPU)
    python ollama_client.py -g 0 "Tell me a story"
    
    # Different model with custom GPU layers and verbose output
    python ollama_client.py -m qwen3-coder -g 50 -v "Explain quantum computing"
"""

import argparse
import json
import sys
from typing import Optional

import requests

OLLAMA_HOST = "http://localhost:11434"  # Change if your server listens elsewhere


def _print_verbose_stats(data: dict) -> None:
    """Print verbose statistics from Ollama response."""
    print("\n" + "=" * 70, file=sys.stderr)
    print("OLLAMA VERBOSE OUTPUT", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    
    # Model information
    if "model" in data:
        print(f"Model: {data['model']}", file=sys.stderr)
    
    # Timing information
    total_duration = data.get("total_duration", 0) / 1e9  # Convert nanoseconds to seconds
    load_duration = data.get("load_duration", 0) / 1e9
    prompt_eval_duration = data.get("prompt_eval_duration", 0) / 1e9
    eval_duration = data.get("eval_duration", 0) / 1e9
    
    print(f"\nTiming:", file=sys.stderr)
    print(f"  Total Duration:         {total_duration:.3f}s", file=sys.stderr)
    print(f"  Model Load Duration:    {load_duration:.3f}s", file=sys.stderr)
    print(f"  Prompt Eval Duration:   {prompt_eval_duration:.3f}s", file=sys.stderr)
    print(f"  Generation Duration:    {eval_duration:.3f}s", file=sys.stderr)
    
    # Token counts
    prompt_eval_count = data.get("prompt_eval_count", 0)
    eval_count = data.get("eval_count", 0)
    
    print(f"\nTokens:", file=sys.stderr)
    print(f"  Prompt Tokens:          {prompt_eval_count}", file=sys.stderr)
    print(f"  Generated Tokens:       {eval_count}", file=sys.stderr)
    print(f"  Total Tokens:           {prompt_eval_count + eval_count}", file=sys.stderr)
    
    # Performance metrics
    if prompt_eval_duration > 0 and prompt_eval_count > 0:
        prompt_tokens_per_sec = prompt_eval_count / prompt_eval_duration
        print(f"  Prompt Eval Speed:      {prompt_tokens_per_sec:.1f} tokens/s", file=sys.stderr)
    
    if eval_duration > 0 and eval_count > 0:
        gen_tokens_per_sec = eval_count / eval_duration
        print(f"  Generation Speed:       {gen_tokens_per_sec:.1f} tokens/s", file=sys.stderr)
    
    # Done reason
    if "done_reason" in data:
        print(f"\nCompletion Reason: {data['done_reason']}", file=sys.stderr)
    
    # Context size
    if "context" in data:
        print(f"Context Size: {len(data['context'])} tokens", file=sys.stderr)
    
    print("=" * 70, file=sys.stderr)


def generate(prompt: str,
             model: str = "gpt-oss:20b",
             stream: bool = True,
             timeout: int = 300,
             num_gpu: Optional[int] = 99,
             verbose: bool = False) -> Optional[str]:
    """
    Send a prompt to the Ollama server and return the generated text.

    Parameters
    ----------
    prompt : str
        The text you want the model to continue or answer.
    model : str, optional
        Model name registered in your Ollama instance (default: "gpt-oss:20b").
    stream : bool, optional
        If True, the server will stream partial results and the function
        will print them incrementally instead of waiting for the full 
response (default: True).
    timeout : int, optional
        Timeout in seconds for the HTTP request (default: 300 seconds).
    num_gpu : int, optional
        Number of GPU layers to use. 99 = use all GPU layers (full GPU),
        0 = use CPU only, 1+ = number of layers to offload to GPU
        (default: 99 for full GPU usage).
    verbose : bool, optional
        If True, display additional statistics like timing, token counts,
        and other metadata from Ollama (default: False).

    Returns
    -------
    str or None
        The final generated text (if `stream` is False).  When `stream` is
        True the function prints the output in real time and returns None.
    """
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
    }
    
    # Add GPU control options if specified
    if num_gpu is not None:
        payload["options"] = {"num_gpu": num_gpu}

    try:
        # If streaming, we need to handle the response as a stream of JSON chunks
        if stream:
            last_data = None
            with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    # Ollama streams lines that look like: {"response":"..."}\n
                    data = json.loads(line.decode("utf-8"))
                    # The "response" key contains the chunk that was just produced
                    chunk = data.get("response", "")
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    # Keep the last data object to get statistics
                    if data.get("done", False):
                        last_data = data
            print()  # newline after streaming ends
            
            # Display verbose statistics if requested
            if verbose and last_data:
                _print_verbose_stats(last_data)
            
            return None
        else:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            
            # Display verbose statistics if requested
            if verbose:
                _print_verbose_stats(data)
            
            return data.get("response", "")
    except requests.exceptions.RequestException as e:
        print(f"[Error] Could not reach Ollama server: {e}", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"[Error] Invalid JSON from server: {e}", file=sys.stderr)
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a prompt to the local Ollama server and print the output."
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="The prompt to send to the LLM.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gpt-oss:20b",
        help="Model name registered with Ollama (default: gpt-oss:20b).",
    )
    parser.add_argument(
        "-s",
        "--stream",
        action="store_true",
        default=True,
        help="Stream the response back to stdout character-by-character (default: True).",
    )
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Disable streaming and wait for full response.",
    )
    parser.add_argument(
        "-g",
        "--num-gpu",
        type=int,
        default=99,
        help="Number of GPU layers to use (99=all GPU, 0=CPU only, default: 99).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show verbose output including timing, token counts, and performance metrics.",
    )
    return parser.parse_args()

    
def main() -> None:
    args = parse_args()
    if args.verbose:
        print(args, file=sys.stderr)
    result = generate(args.prompt, model=args.model, stream=args.stream, 
                     num_gpu=args.num_gpu, verbose=args.verbose)
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()
