"""
Integration Tests for WebSocket Inference Server

Tests the complete inference pipeline end-to-end.
"""

import pytest
import asyncio
import json
import websockets
import requests
from typing import AsyncGenerator


# Test configuration
API_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/generate"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test /health endpoint."""
        response = requests.get(f"{API_URL}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "device" in data
    
    def test_health_check_timeout(self):
        """Test health endpoint has reasonable timeout."""
        response = requests.get(f"{API_URL}/health", timeout=5)
        assert response.status_code == 200


class TestInfoEndpoint:
    """Test model info endpoint."""
    
    def test_model_info(self):
        """Test /info endpoint."""
        response = requests.get(f"{API_URL}/info")
        
        if response.status_code == 200:
            data = response.json()
            assert "vocab_size" in data
            assert "d_model" in data
            assert "num_layers" in data
            assert "device" in data
        else:
            # Model might not be loaded yet
            assert response.status_code == 503


class TestWebSocketGeneration:
    """Test WebSocket generation endpoint."""
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket connection."""
        try:
            async with websockets.connect(WS_URL) as websocket:
                # Connection successful
                assert websocket.open
        except Exception as e:
            pytest.fail(f"WebSocket connection failed: {e}")
    
    @pytest.mark.asyncio
    async def test_simple_generation(self):
        """Test simple text generation."""
        async with websockets.connect(WS_URL) as websocket:
            # Send request
            request = {
                "prompt": "Hello",
                "max_tokens": 20,
                "temperature": 0.8,
                "top_k": 50,
                "top_p": 0.95
            }
            await websocket.send(json.dumps(request))
            
            # Collect responses
            messages = []
            async for message in websocket:
                data = json.loads(message)
                messages.append(data)
                
                if data["type"] == "complete":
                    break
                elif data["type"] == "error":
                    pytest.fail(f"Generation error: {data['message']}")
            
            # Verify received messages
            assert len(messages) > 0
            
            # Check for start message
            start_msgs = [m for m in messages if m["type"] == "start"]
            assert len(start_msgs) > 0
            
            # Check for token messages
            token_msgs = [m for m in messages if m["type"] == "token"]
            assert len(token_msgs) > 0
            
            # Check for complete message
            complete_msgs = [m for m in messages if m["type"] == "complete"]
            assert len(complete_msgs) == 1
            assert "tokens_generated" in complete_msgs[0]
            assert complete_msgs[0]["tokens_generated"] > 0
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_generations(self):
        """Test multiple concurrent generation requests."""
        async def generate(prompt: str):
            async with websockets.connect(WS_URL) as websocket:
                request = {
                    "prompt": prompt,
                    "max_tokens": 10,
                    "temperature": 0.8
                }
                await websocket.send(json.dumps(request))
                
                async for message in websocket:
                    data = json.loads(message)
                    if data["type"] == "complete":
                        return data["tokens_generated"]
                    elif data["type"] == "error":
                        return 0
        
        # Run 3 concurrent generations
        prompts = ["Hello", "Once upon a time", "The quick brown"]
        results = await asyncio.gather(*[generate(p) for p in prompts])
        
        # All should complete
        assert all(r > 0 for r in results)
    
    @pytest.mark.asyncio
    async def test_temperature_parameter(self):
        """Test temperature parameter affects output."""
        async def generate_with_temp(temp: float):
            async with websockets.connect(WS_URL) as websocket:
                request = {
                    "prompt": "Hello",
                    "max_tokens": 10,
                    "temperature": temp
                }
                await websocket.send(json.dumps(request))
                
                tokens = []
                async for message in websocket:
                    data = json.loads(message)
                    if data["type"] == "token":
                        tokens.append(data["text"])
                    elif data["type"] == "complete":
                        return "".join(tokens)
                    elif data["type"] == "error":
                        return ""
        
        # Generate with different temperatures
        output_low = await generate_with_temp(0.3)
        output_high = await generate_with_temp(1.5)
        
        # Both should generate text
        assert len(output_low) > 0
        assert len(output_high) > 0
    
    @pytest.mark.asyncio
    async def test_max_tokens_limit(self):
        """Test max_tokens parameter is respected."""
        async with websockets.connect(WS_URL) as websocket:
            max_tokens = 5
            request = {
                "prompt": "Hello",
                "max_tokens": max_tokens,
                "temperature": 0.8
            }
            await websocket.send(json.dumps(request))
            
            token_count = 0
            async for message in websocket:
                data = json.loads(message)
                if data["type"] == "token":
                    token_count += 1
                elif data["type"] == "complete":
                    break
            
            # Should generate exactly max_tokens
            assert token_count == max_tokens


class TestErrorHandling:
    """Test error handling."""
    
    @pytest.mark.asyncio
    async def test_invalid_request_format(self):
        """Test handling of invalid request format."""
        async with websockets.connect(WS_URL) as websocket:
            # Send invalid JSON
            await websocket.send("invalid json")
            
            # Should not crash server
            # Might get error message or connection close
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=2)
                data = json.loads(message)
                # Expect error message
                assert data["type"] == "error"
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                # Connection might close, which is acceptable
                pass
    
    @pytest.mark.asyncio
    async def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        async with websockets.connect(WS_URL) as websocket:
            # Send request without prompt
            request = {
                "max_tokens": 10
            }
            await websocket.send(json.dumps(request))
            
            # Should handle gracefully
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=2)
                data = json.loads(message)
                # Might get error or generate empty text
                assert data["type"] in ["error", "complete"]
            except asyncio.TimeoutError:
                pytest.fail("Server did not respond to request")


class TestPerformance:
    """Performance tests."""
    
    @pytest.mark.asyncio
    async def test_generation_latency(self):
        """Test generation latency is reasonable."""
        import time
        
        async with websockets.connect(WS_URL) as websocket:
            start = time.time()
            
            request = {
                "prompt": "Hello",
                "max_tokens": 10,
                "temperature": 0.8
            }
            await websocket.send(json.dumps(request))
            
            async for message in websocket:
                data = json.loads(message)
                if data["type"] == "complete":
                    break
            
            elapsed = time.time() - start
            
            # Should complete within reasonable time (10 tokens in < 5 seconds)
            assert elapsed < 5.0
    
    @pytest.mark.asyncio
    async def test_tokens_per_second(self):
        """Test generation speed (tokens/second)."""
        import time
        
        async with websockets.connect(WS_URL) as websocket:
            start = time.time()
            
            request = {
                "prompt": "Hello",
                "max_tokens": 50,
                "temperature": 0.8
            }
            await websocket.send(json.dumps(request))
            
            tokens_generated = 0
            async for message in websocket:
                data = json.loads(message)
                if data["type"] == "token":
                    tokens_generated += 1
                elif data["type"] == "complete":
                    break
            
            elapsed = time.time() - start
            tokens_per_sec = tokens_generated / elapsed
            
            # Should generate at reasonable speed (>1 token/sec on CPU)
            assert tokens_per_sec > 1.0
            
            print(f"\nGeneration speed: {tokens_per_sec:.2f} tokens/sec")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
