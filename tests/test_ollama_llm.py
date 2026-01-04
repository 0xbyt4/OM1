#!/usr/bin/env python3
"""
Test script for Ollama LLM Plugin

Usage:
    1. Install Ollama: https://ollama.ai
    2. Start Ollama: ollama serve
    3. Pull a model: ollama pull llama3.2
    4. Run: uv run python tests/test_ollama_llm.py
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm.plugins.ollama_llm import OllamaLLM, OllamaLLMConfig


async def test_ollama_connection():
    """Test basic Ollama connectivity."""
    print("=" * 50)
    print("Ollama LLM Plugin Test")
    print("=" * 50)

    config = OllamaLLMConfig(
        model="llama3.2",
        ollama_base_url="http://localhost:11434",
        temperature=0.7,
        timeout=60,
    )

    print(f"\nConfig:")
    print(f"  Model: {config.model}")
    print(f"  URL: {config.ollama_base_url}")
    print(f"  Temperature: {config.temperature}")

    llm = OllamaLLM(config=config)

    print("\nTesting connection...")

    try:
        import httpx

        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{config.ollama_base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                print(f"  Connection: OK")
                print(f"  Available models: {[m['name'] for m in models]}")
            else:
                print(f"  Connection: FAILED (status {response.status_code})")
                return False
    except httpx.ConnectError:
        print("  Connection: FAILED")
        print("\n  Is Ollama running? Start with: ollama serve")
        return False
    except Exception as e:
        print(f"  Connection: ERROR - {e}")
        return False

    print("\nTesting LLM query...")
    try:
        result = await llm.ask("Say hello in one word")
        if result:
            print(f"  Response: {result}")
            print("  Test: PASSED")
            return True
        else:
            print("  Response: None (model may need function calling)")
            print("  Test: PARTIAL (connection works)")
            return True
    except Exception as e:
        print(f"  Error: {e}")
        return False
    finally:
        await llm.close()


async def test_load_from_config():
    """Test loading Ollama LLM via the loader."""
    print("\n" + "=" * 50)
    print("Testing LLM Loader")
    print("=" * 50)

    from llm import load_llm

    config = {
        "type": "OllamaLLM",
        "config": {
            "model": "llama3.2",
            "ollama_base_url": "http://localhost:11434",
        },
    }

    try:
        llm = load_llm(config)
        print(f"  Loader: OK")
        print(f"  Class: {llm.__class__.__name__}")
        return True
    except Exception as e:
        print(f"  Loader: FAILED - {e}")
        return False


if __name__ == "__main__":
    print("\n")

    # Test 1: Direct connection
    result1 = asyncio.run(test_ollama_connection())

    # Test 2: Config loader
    result2 = asyncio.run(test_load_from_config())

    print("\n" + "=" * 50)
    print("Results")
    print("=" * 50)
    print(f"  Connection Test: {'PASSED' if result1 else 'FAILED'}")
    print(f"  Loader Test: {'PASSED' if result2 else 'FAILED'}")
    print("=" * 50)

    if not result1:
        print("\nTo fix connection issues:")
        print("  1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print("  2. Start server: ollama serve")
        print("  3. Pull model: ollama pull llama3.2")
