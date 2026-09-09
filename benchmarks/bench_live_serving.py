import json
import time
import urllib.request
import urllib.error
import sys

SERVER_URL = "http://127.0.0.1:30000"

def wait_for_server(timeout=300):
    print(f"Waiting for SGLang server at {SERVER_URL} (timeout {timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{SERVER_URL}/v1/models")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    print("SGLang server is online and responding!")
                    return True
        except Exception:
            time.sleep(3)
    print("Server failed to come online within timeout.")
    return False

def test_chat_completion(prompt, max_tokens=64, temperature=0.0):
    url = f"{SERVER_URL}/v1/chat/completions"
    payload = {
        "model": "default",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw_bytes = resp.read()
        elapsed = time.perf_counter() - t0
        body = json.loads(raw_bytes.decode("utf-8"))
    
    usage = body.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    msg = body["choices"][0]["message"]
    content = msg.get("content") or msg.get("reasoning_content") or ""
    
    decode_tok_per_sec = (completion_tokens / elapsed) if elapsed > 0 else 0.0
    return {
        "elapsed_sec": round(elapsed, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_sec": round(decode_tok_per_sec, 2),
        "content_preview": content[:120].strip()
    }

def test_streaming_ttft(prompt, max_tokens=32):
    url = f"{SERVER_URL}/v1/chat/completions"
    payload = {
        "model": "default",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    tokens_text = []
    stream_completion_tokens = 0
    
    first_chunk_time = None
    with urllib.request.urlopen(req, timeout=60) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                t_now = time.perf_counter()
                chunks += 1
                if first_chunk_time is None:
                    first_chunk_time = t_now - t0
                try:
                    chunk_obj = json.loads(line[6:])
                    usage = chunk_obj.get("usage")
                    if usage and "completion_tokens" in usage:
                        stream_completion_tokens = int(usage["completion_tokens"])
                    choices = chunk_obj.get("choices", [{}])
                    delta = choices[0].get("delta", {}) if choices else {}
                    content_piece = (
                        delta.get("content")
                        or delta.get("reasoning_content")
                        or delta.get("text")
                        or (choices[0].get("text") if choices else None)
                        or ""
                    )
                    if content_piece:
                        if ttft is None:
                            ttft = t_now - t0
                        tokens_text.append(content_piece)
                except Exception:
                    pass
    if ttft is None and chunks > 0:
        ttft = first_chunk_time or (time.perf_counter() - t0)
    total_time = time.perf_counter() - t0
    decode_time = total_time - (ttft or 0)
    full_text = "".join(tokens_text)
    if stream_completion_tokens > 0:
        num_tokens = stream_completion_tokens
    elif len(full_text) > 0:
        # Standard fallback approximation: 1 token ~= 4 characters or max chunks
        num_tokens = max(round(len(full_text) / 4.0), chunks)
    else:
        num_tokens = max(chunks - 1, 1)
    tok_per_sec = (num_tokens / decode_time) if decode_time > 0 else 0.0
    
    return {
        "ttft_ms": round((ttft or 0) * 1000, 2),
        "total_time_sec": round(total_time, 4),
        "decode_time_sec": round(decode_time, 4),
        "streamed_chunks": chunks,
        "tokens_generated": num_tokens,
        "decode_tok_per_sec": round(tok_per_sec, 2),
        "content_preview": full_text[:120].strip()
    }

def run_benchmarks():
    if not wait_for_server():
        sys.exit(1)
    
    print("\n================ LIVE SGLANG EXL3 END-TO-END BENCHMARK ================")
    
    # Test 1: Cold prefill & decode
    print("\n--- 1. Cold Request (Fresh Prompt) ---")
    cold_prompt = "Explain quantum superposition and entanglement in simple terms for a computer scientist."
    cold_res = test_chat_completion(cold_prompt, max_tokens=64)
    print(f"Latency: {cold_res['elapsed_sec']} s | Tokens: {cold_res['completion_tokens']} | Speed: {cold_res['tokens_per_sec']} tok/s")
    print(f"Output: {cold_res['content_preview']}...")
    
    # Test 2: Streaming TTFT
    print("\n--- 2. Streaming Request (TTFT & Decode Rate) ---")
    stream_prompt = "Compare RISC-V and ARM architectures focusing on instruction encoding and pipelining."
    stream_res = test_streaming_ttft(stream_prompt, max_tokens=32)
    print(f"TTFT: {stream_res['ttft_ms']} ms | Chunks: {stream_res['streamed_chunks']} | Decode Speed: {stream_res['decode_tok_per_sec']} tok/s")
    
    # Test 3: Warm prefix request (Testing RadixAttention + Hadamard Cache Hit)
    print("\n--- 3. Warm Prefix Request (Radix & Hadamard Cache Reuse) ---")
    prefix = "You are a senior systems architect specializing in AI accelerators and high-performance computing. " * 10
    warm_prompt_1 = prefix + "What is the memory bandwidth of NVLink-C2C on GB10?"
    _ = test_chat_completion(warm_prompt_1, max_tokens=16)  # Warm up prefix
    
    warm_res = test_streaming_ttft(warm_prompt_1 + " Give a concise summary.", max_tokens=32)
    print(f"Warm TTFT: {warm_res['ttft_ms']} ms | Decode Speed: {warm_res['decode_tok_per_sec']} tok/s")
    
    # Test 4: Sustained 128-token Decode Throughput
    print("\n--- 4. Sustained 128-Token Decode ---")
    sustained_prompt = "Write a comprehensive essay on the history and architecture of the Internet."
    sustained_res = test_chat_completion(sustained_prompt, max_tokens=128)
    print(f"Latency: {sustained_res['elapsed_sec']} s | Tokens: {sustained_res['completion_tokens']} | Speed: {sustained_res['tokens_per_sec']} tok/s")
    
    # Test 5: Sustained 256-token Decode Throughput
    print("\n--- 5. Sustained 256-Token Decode ---")
    sustained_256_prompt = "Write an exhaustive technical treatise on quantum error correction, surface codes, and logical qubits."
    sustained_256_res = test_chat_completion(sustained_256_prompt, max_tokens=256)
    print(f"Latency: {sustained_256_res['elapsed_sec']} s | Tokens: {sustained_256_res['completion_tokens']} | Speed: {sustained_256_res['tokens_per_sec']} tok/s")

    # Test 6: Sustained 512-token Decode Throughput
    print("\n--- 6. Sustained 512-Token Decode ---")
    sustained_512_prompt = "Provide a deep dive into modern microprocessor architecture, comparing out-of-order execution, speculative execution vulnerabilities, and GPU SIMD/SIMT models in exhaustive detail."
    sustained_512_res = test_chat_completion(sustained_512_prompt, max_tokens=512)
    print(f"Latency: {sustained_512_res['elapsed_sec']} s | Tokens: {sustained_512_res['completion_tokens']} | Speed: {sustained_512_res['tokens_per_sec']} tok/s")

    summary = {
        "cold_request": cold_res,
        "streaming_request": stream_res,
        "warm_prefix_request": warm_res,
        "sustained_128_request": sustained_res,
        "sustained_256_request": sustained_256_res,
        "sustained_512_request": sustained_512_res,
    }
    
    print("\n================ FINAL EMPIRICAL SERVING RECEIPTS ================")
    print(json.dumps(summary, indent=2))
    print("==================================================================\n")

if __name__ == "__main__":
    run_benchmarks()
