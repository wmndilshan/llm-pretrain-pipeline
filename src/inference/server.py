"""
Unified WebSocket Inference Server

Features:
1. Real-time text generation with WebSocket streaming
2. Optional adaptive hardware detection (GPU/MPS/CPU)
3. Optional model quantization for CPU inference
4. KV caching for efficient generation
5. Multiple concurrent connections support

Usage:
    # Start with default settings
    uvicorn src.inference.server:app --host 0.0.0.0 --port 8000

    # Or run with adaptive mode
    ADAPTIVE_MODE=1 uvicorn src.inference.server:app --host 0.0.0.0 --port 8000
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import json
import asyncio

import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.tokenization import TokenizerManager
from src.inference.loading import load_checkpoint_model


# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.95
    stream: bool = True


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    finish_reason: str


class ModelInfo(BaseModel):
    model_path: str
    vocab_size: int
    d_model: int
    num_layers: int
    num_heads: int
    max_seq_length: int
    device: str
    quantized: bool = False
    adaptive_mode: bool = False


# Hardware detection and optimization
class HardwareDetector:
    """Detect available hardware and configure accordingly."""

    @staticmethod
    def detect_device() -> str:
        """Detect best available device. Priority: CUDA > MPS > CPU"""
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("MPS (Apple Silicon) detected")
        else:
            device = "cpu"
            logger.info("No GPU detected, using CPU")
        return device

    @staticmethod
    def should_quantize(device: str, model_size_mb: float) -> bool:
        """Determine if model should be quantized."""
        return device == "cpu"

    @staticmethod
    def get_optimal_batch_size(device: str) -> int:
        """Get optimal batch size for device."""
        if device == "cuda":
            return 8
        elif device == "mps":
            return 4
        return 1


class ModelQuantizer:
    """Quantize models for efficient CPU inference."""

    @staticmethod
    def quantize_dynamic(model: nn.Module) -> nn.Module:
        """Apply dynamic quantization for CPU inference."""
        logger.info("Applying dynamic quantization...")
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )
        logger.info("Model quantized (dynamic INT8)")
        return quantized_model

    @staticmethod
    def get_model_size(model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024**2)


# FastAPI app
app = FastAPI(
    title="LLM Inference Server",
    description="WebSocket-based inference server for transformer language models",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model: Optional[nn.Module] = None
tokenizer: Optional[TokenizerManager] = None
device: str = "cpu"
model_config: Dict[str, Any] = {}
is_quantized: bool = False
adaptive_mode: bool = os.environ.get('ADAPTIVE_MODE', '').lower() in ('1', 'true', 'yes')


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")

    async def send_message(self, websocket: WebSocket, message: dict):
        await websocket.send_json(message)


manager = ConnectionManager()


def load_model_and_tokenizer(
    model_path: str,
    tokenizer_path: str,
    device_name: str = "auto",
    force_quantize: bool = False
):
    """Load model and tokenizer on startup."""
    global model, tokenizer, device, model_config, is_quantized, adaptive_mode

    logger.info("Loading model and tokenizer...")

    # Setup device
    if device_name == "auto" or adaptive_mode:
        device = HardwareDetector.detect_device()
    elif device_name == "cuda" and torch.cuda.is_available():
        device = "cuda"
    elif device_name == "mps" and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(f"Using device: {device}")

    model, runtime_model_config, _ = load_checkpoint_model(model_path, device="cpu")
    model_config = runtime_model_config.to_dict()

    # Get model size
    model_size_mb = ModelQuantizer.get_model_size(model)
    logger.info(f"Model size: {model_size_mb:.1f}MB")
    logger.info(f"Parameters: {model.count_parameters():,}")

    # Quantize if needed (adaptive mode or forced)
    if force_quantize or (adaptive_mode and HardwareDetector.should_quantize(device, model_size_mb)):
        model = ModelQuantizer.quantize_dynamic(model)
        is_quantized = True
        # Quantized models stay on CPU
    else:
        model = model.to(device)

    model.eval()

    # Load tokenizer
    tokenizer = TokenizerManager.from_pretrained(tokenizer_path)

    logger.info("Model and tokenizer loaded successfully")
    logger.info(f"  Vocab size: {tokenizer.get_vocab_size():,}")
    if adaptive_mode:
        logger.info(f"  Adaptive mode: enabled")
        logger.info(f"  Quantized: {is_quantized}")


@torch.no_grad()
async def generate_streaming(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    websocket: WebSocket
):
    """Generate text with streaming output."""
    if model is None or tokenizer is None:
        await manager.send_message(websocket, {
            'type': 'error',
            'message': 'Model not loaded'
        })
        return

    try:
        # Encode prompt
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([input_ids], dtype=torch.long)

        # Move to device only if not quantized
        if not is_quantized:
            input_ids = input_ids.to(device)

        # Send start message
        await manager.send_message(websocket, {
            'type': 'start',
            'prompt': prompt,
            'device': device,
            'quantized': is_quantized
        })

        # Generate tokens one at a time
        cache = None
        generated_tokens = 0

        for step in range(max_tokens):
            # Get predictions
            if cache is None:
                idx_cond = input_ids
            else:
                idx_cond = input_ids[:, -1:]

            logits, _, cache = model(idx_cond, cache=cache)

            # Get last token logits
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Apply top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Decode and send token
            token_text = tokenizer.decode([next_token.item()])

            await manager.send_message(websocket, {
                'type': 'token',
                'text': token_text,
                'token_id': next_token.item()
            })

            generated_tokens += 1

            # Adaptive delay (faster on GPU)
            delay = 0.001 if device == "cuda" else 0.01
            await asyncio.sleep(delay)

        # Send completion message
        full_text = tokenizer.decode(input_ids[0].tolist())

        await manager.send_message(websocket, {
            'type': 'complete',
            'full_text': full_text,
            'tokens_generated': generated_tokens,
            'finish_reason': 'max_tokens',
            'device': device,
            'quantized': is_quantized
        })

    except Exception as e:
        logger.error(f"Generation error: {e}")
        await manager.send_message(websocket, {
            'type': 'error',
            'message': str(e)
        })


# REST endpoints
@app.get("/")
async def root():
    """Root endpoint with simple test UI."""
    adaptive_status = f'<span class="status adaptive">ADAPTIVE</span>' if adaptive_mode else ''
    quantized_status = f'<span class="status quantized">QUANTIZED</span>' if is_quantized else ''

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LLM Inference Server</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
            }}
            .container {{
                border: 1px solid #ccc;
                padding: 20px;
                border-radius: 5px;
            }}
            .hardware-info {{
                background-color: #f0f0f0;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .status {{
                display: inline-block;
                padding: 5px 10px;
                border-radius: 3px;
                margin-right: 10px;
                font-weight: bold;
            }}
            .status.gpu {{ background-color: #4CAF50; color: white; }}
            .status.cpu {{ background-color: #ff9800; color: white; }}
            .status.mps {{ background-color: #9C27B0; color: white; }}
            .status.quantized {{ background-color: #2196F3; color: white; }}
            .status.adaptive {{ background-color: #00BCD4; color: white; }}
            textarea {{
                width: 100%;
                height: 100px;
                margin: 10px 0;
            }}
            button {{
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border: none;
                cursor: pointer;
                border-radius: 4px;
            }}
            button:hover {{
                background-color: #45a049;
            }}
            #output {{
                margin-top: 20px;
                padding: 15px;
                background-color: #f4f4f4;
                border-radius: 4px;
                min-height: 100px;
                white-space: pre-wrap;
            }}
            .status-text {{
                color: #666;
                font-size: 14px;
                margin-top: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>LLM Inference Server</h1>

            <div class="hardware-info">
                <h3>Hardware Configuration</h3>
                <span class="status {device}">{device.upper()}</span>
                {quantized_status}
                {adaptive_status}
                <p>WebSocket-based real-time text generation</p>
            </div>

            <textarea id="prompt" placeholder="Enter your prompt here...">Once upon a time</textarea>

            <div>
                <label>Max Tokens: <input type="number" id="maxTokens" value="100" min="1" max="500"></label>
                <label>Temperature: <input type="number" id="temperature" value="0.8" min="0.1" max="2.0" step="0.1"></label>
            </div>

            <button onclick="generate()">Generate</button>

            <div class="status-text" id="status">Ready</div>
            <div id="output"></div>
        </div>

        <script>
            let ws = null;

            function generate() {{
                const prompt = document.getElementById('prompt').value;
                const maxTokens = parseInt(document.getElementById('maxTokens').value);
                const temperature = parseFloat(document.getElementById('temperature').value);

                document.getElementById('output').textContent = prompt;
                document.getElementById('status').textContent = 'Connecting...';

                ws = new WebSocket(`ws://${{window.location.host}}/ws/generate`);

                ws.onopen = () => {{
                    document.getElementById('status').textContent = 'Generating...';
                    ws.send(JSON.stringify({{
                        prompt: prompt,
                        max_tokens: maxTokens,
                        temperature: temperature,
                        top_k: 50,
                        top_p: 0.95
                    }}));
                }};

                ws.onmessage = (event) => {{
                    const message = JSON.parse(event.data);
                    const output = document.getElementById('output');

                    if (message.type === 'token') {{
                        output.textContent += message.text;
                    }} else if (message.type === 'complete') {{
                        document.getElementById('status').textContent =
                            `Complete (${{message.tokens_generated}} tokens, device: ${{message.device}})`;
                        ws.close();
                    }} else if (message.type === 'error') {{
                        document.getElementById('status').textContent =
                            `Error: ${{message.message}}`;
                        ws.close();
                    }}
                }};

                ws.onerror = (error) => {{
                    document.getElementById('status').textContent = 'Connection error';
                    console.error(error);
                }};
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
        "quantized": is_quantized,
        "adaptive_mode": adaptive_mode,
        "optimal_batch_size": HardwareDetector.get_optimal_batch_size(device)
    }


@app.get("/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfo(
        model_path="loaded",
        vocab_size=model_config.get('vocab_size', 0),
        d_model=model_config.get('d_model', 0),
        num_layers=model_config.get('num_layers', 0),
        num_heads=model_config.get('num_heads', 0),
        max_seq_length=model_config.get('max_seq_length', 0),
        device=device,
        quantized=is_quantized,
        adaptive_mode=adaptive_mode
    )


# WebSocket endpoint
@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """WebSocket endpoint for streaming generation."""
    await manager.connect(websocket)

    try:
        while True:
            # Receive request
            data = await websocket.receive_text()
            request = json.loads(data)

            # Validate request
            prompt = request.get('prompt', '')
            max_tokens = request.get('max_tokens', 100)
            temperature = request.get('temperature', 0.8)
            top_k = request.get('top_k', 50)
            top_p = request.get('top_p', 0.95)

            # Generate with streaming
            await generate_streaming(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                websocket=websocket
            )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    model_path = os.environ.get('MODEL_PATH', '/app/models/best_model.pt')
    tokenizer_path = os.environ.get('TOKENIZER_PATH', '/app/models/tokenizer.json')

    if Path(model_path).exists() and Path(tokenizer_path).exists():
        load_model_and_tokenizer(model_path, tokenizer_path)
    else:
        logger.warning("Model or tokenizer not found. Server started without model.")
        logger.info(f"Expected model at: {model_path}")
        logger.info(f"Expected tokenizer at: {tokenizer_path}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.inference.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
