import os
import asyncio
import httpx
from dotenv import load_dotenv

# =====================================================================
# 1. ENVIRONMENT & GLOBALS SETUP
# =====================================================================
load_dotenv()
API_KEY = os.getenv("NVIDIA_API_KEY")

if not API_KEY:
    raise ValueError("❌ Critical Error: NVIDIA_API_KEY is missing from your .env file.")

# Define global headers immediately so all functions can read them safely
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

# =====================================================================
# 2. ENDPOINT MAPPINGS
# =====================================================================
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Chat/Reasoning and Embeddings live on the new unified integration gateway
NVIDIA_CHAT_URL = f"{NVIDIA_BASE_URL}/chat/completions"
NVIDIA_EMBED_URL = f"{NVIDIA_BASE_URL}/embeddings"

# ⚠️ Reranker remains hosted on the dedicated legacy retrieval subdomain
NVIDIA_RERANK_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"

# =====================================================================
# 3. DIAGNOSTIC TEST ROUTINES
# =====================================================================

def test_chat_and_reasoning():
    print("🎬 Testing Chat/Reasoning Endpoint (Llama 4 Maverick)...")
    payload = {
        "model": "meta/llama-4-maverick-17b-128e-instruct",
        "messages": [{"role": "user", "content": "Hello! Confirm operational readiness."}],
        "max_tokens": 15,
        "temperature": 0.1
    }
    try:
        response = httpx.post(NVIDIA_CHAT_URL, headers=HEADERS, json=payload, timeout=10.0)
        if response.status_code == 200:
            text = response.json()["choices"][0]["message"]["content"].strip()
            print(f"  ✅ Chat Success! Response: \"{text}\"")
        else:
            print(f"  ❌ Chat Failed ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"  ❌ Chat Exception: {e}")


def test_chunk_embedding():
    print("🔢 Testing Chunk Embedding Endpoint (nv-embed-v1)...")
    payload = {
        "model": "nvidia/nv-embed-v1",
        "input": ["Diagnostic verification for dynamic vector database chunking."],
        "input_type": "passage",
        "encoding_format": "float"
    }
    try:
        response = httpx.post(NVIDIA_EMBED_URL, headers=HEADERS, json=payload, timeout=10.0)
        if response.status_code == 200:
            embedding_vector = response.json()["data"][0]["embedding"]
            print(f"  ✅ Embedding Success! Generated vector array of size: {len(embedding_vector)}")
        else:
            print(f"  ❌ Embedding Failed ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"  ❌ Embedding Exception: {e}")


def test_vector_reranking():
    print("🎯 Testing Vector Reranking Endpoint (rerank-qa-mistral-4b)...")
    # Note: Payload key must be "passages" for this endpoint
    payload = {
        "model": "nvidia/rerank-qa-mistral-4b",
        "query": {"text": "What is the status?"},
        "passages": [
            {"text": "The operational framework status is fully optimized and normal."},
            {"text": "An unrelated snippet containing background noise and mismatch elements."}
        ]
    }
    try:
        response = httpx.post(NVIDIA_RERANK_URL, headers=HEADERS, json=payload, timeout=10.0)
        if response.status_code == 200:
            print("  ✅ Reranking Success!")
            scores = response.json().get("rankings", [])
            if scores:
                print(f"     Top Parsed Logit Score: {scores[0]['logit']}")
        else:
            print(f"  ❌ Reranking Failed ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"  ❌ Reranking Exception: {e}")


# =====================================================================
# 4. EXECUTION ENGINE
# =====================================================================
if __name__ == "__main__":
    print("🚀 Starting Unified Pipeline Endpoint Diagnostics...\n")
    
    test_chat_and_reasoning()
    print("---")
    test_chunk_embedding()
    print("---")
    test_vector_reranking()
    
    print("\n🎉 Diagnostics Complete. If all paths show green checkmarks, proceed with the Streamlit app assembly.")