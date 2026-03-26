"""
Live streaming inference from the WebSocket server (CLI, like real LLM terminals).
Usage:
  Start server: python inference_server.py  (or uvicorn)
  Stream:       python scripts/stream_inference.py --prompt "Once upon a time"
  With server:  python scripts/stream_inference.py --url ws://localhost:8000/ws/generate --prompt "Hello"
"""
import argparse
import json
import sys

try:
    import websocket
except ImportError:
    print("[FAIL] Install: pip install websocket-client", file=sys.stderr)
    sys.exit(1)


def stream_generate(url: str, prompt: str, max_tokens: int = 100, temperature: float = 0.8) -> int:
    ws = websocket.create_connection(url)
    try:
        ws.send(json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }))
        tokens = 0
        while True:
            raw = ws.recv()
            msg = json.loads(raw)
            t = msg.get("type")
            if t == "token":
                sys.stdout.write(msg.get("text", ""))
                sys.stdout.flush()
                tokens += 1
            elif t == "complete":
                sys.stdout.write("\n")
                sys.stdout.flush()
                return tokens
            elif t == "error":
                print(f"\n[FAIL] {msg.get('message', 'Unknown error')}", file=sys.stderr)
                return -1
    finally:
        ws.close()


def main():
    p = argparse.ArgumentParser(description="Stream LLM output from WebSocket server.")
    p.add_argument("--url", default="ws://localhost:8000/ws/generate", help="WebSocket URL")
    p.add_argument("--prompt", "-p", required=True, help="Prompt text")
    p.add_argument("--max-tokens", "-n", type=int, default=150, help="Max tokens to generate")
    p.add_argument("--temperature", "-t", type=float, default=0.8, help="Sampling temperature")
    args = p.parse_args()

    print(f"[INFO] Connecting to {args.url} ...", file=sys.stderr)
    try:
        n = stream_generate(args.url, args.prompt, args.max_tokens, args.temperature)
        if n >= 0:
            print(f"[INFO] Generated {n} tokens.", file=sys.stderr)
    except Exception as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
