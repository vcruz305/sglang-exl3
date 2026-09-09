import concurrent.futures
import json
import statistics
import time
import urllib.request
import urllib.error
import sys
from typing import List, Dict, Any

SERVER_URL = "http://127.0.0.1:30000"

PROMPTS = [
    "Explain quantum superposition and entanglement in simple terms for a computer scientist.",
    "Compare RISC-V and ARM architectures focusing on instruction encoding and pipelining.",
    "Explain the mechanics of Grace Blackwell NVLink-C2C unified memory and its bandwidth.",
    "Describe the key differences between RadixAttention prefix caching and PagedAttention.",
    "What are the mathematical properties of the Walsh-Hadamard transform in MoE quantization?",
    "How does speculative decoding with MTP multiple token prediction improve decode throughput?",
    "Explain the internal structure of Trellis quantization in ExLlamaV3 with MCG codebooks.",
    "Discuss the trade-offs between dense linear layers and mixture of experts in large language models."
]

def wait_for_server(timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{SERVER_URL}/v1/models")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(3)
    return False

def send_chat_request(prompt: str, max_tokens: int = 64) -> Dict[str, Any]:
    url = f"{SERVER_URL}/v1/chat/completions"
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            elapsed = time.perf_counter() - t0
            body = json.loads(resp.read().decode("utf-8"))
            usage = body.get("usage", {})
            return {
                "success": True,
                "elapsed": elapsed,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "success": False,
            "elapsed": elapsed,
            "error": str(exc),
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

def run_concurrency_level(concurrency: int, requests_per_worker: int = 3, max_tokens: int = 64) -> Dict[str, Any]:
    total_requests = concurrency * requests_per_worker
    tasks = []
    for i in range(total_requests):
        prompt = PROMPTS[i % len(PROMPTS)]
        tasks.append(prompt)
    
    t_start = time.perf_counter()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_chat_request, p, max_tokens) for p in tasks]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
    t_total = time.perf_counter() - t_start
    
    successful = [r for r in results if r["success"]]
    total_completion_tokens = sum(r["completion_tokens"] for r in successful)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in successful)
    latencies_ms = [r["elapsed"] * 1000 for r in successful]
    
    output_tok_per_sec = total_completion_tokens / t_total if t_total > 0 else 0.0
    total_tok_per_sec = (total_prompt_tokens + total_completion_tokens) / t_total if t_total > 0 else 0.0
    
    p50_lat = statistics.median(latencies_ms) if latencies_ms else 0.0
    latencies_sorted = sorted(latencies_ms)
    p95_lat = latencies_sorted[int(len(latencies_sorted) * 0.95)] if latencies_sorted else 0.0
    p99_lat = latencies_sorted[int(len(latencies_sorted) * 0.99)] if latencies_sorted else 0.0
    
    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful_requests": len(successful),
        "total_elapsed_sec": round(t_total, 3),
        "total_completion_tokens": total_completion_tokens,
        "output_tokens_per_sec": round(output_tok_per_sec, 2),
        "total_tokens_per_sec": round(total_tok_per_sec, 2),
        "latency_p50_ms": round(p50_lat, 2),
        "latency_p95_ms": round(p95_lat, 2),
        "latency_p99_ms": round(p99_lat, 2),
    }

def main():
    if not wait_for_server():
        print("Error: SGLang server is not reachable at", SERVER_URL)
        sys.exit(1)
    
    print("\n================ SGLANG EXL3 MULTI-STREAM CONCURRENCY BENCHMARK ================")
    print("Server:", SERVER_URL)
    print("Concurrency levels: [1, 2, 4, 8, 16, 32]")
    print("---------------------------------------------------------------------------------")
    print(" Concurrency | Total Req | Out Tok/s | Total Tok/s | p50 (ms) | p95 (ms) | Speedup vs C1")
    print("-------------+-----------+-----------+-------------+----------+----------+--------------")
    
    concurrency_levels = [1, 2, 4, 8, 16, 32]
    all_results = []
    c1_throughput = 1.0
    
    for c in concurrency_levels:
        res = run_concurrency_level(c, requests_per_worker=3, max_tokens=64)
        if c == 1:
            c1_throughput = max(res["output_tokens_per_sec"], 0.1)
        speedup = res["output_tokens_per_sec"] / c1_throughput
        res["speedup_vs_c1"] = round(speedup, 2)
        all_results.append(res)
        
        print(f" {c:11d} | {res['successful_requests']:9d} | {res['output_tokens_per_sec']:9.2f} | {res['total_tokens_per_sec']:11.2f} | {res['latency_p50_ms']:8.1f} | {res['latency_p95_ms']:8.1f} | {speedup:11.2f}x")
        time.sleep(2)
        
    print("=================================================================================\n")
    print(json.dumps(all_results, indent=2))

if __name__ == "__main__":
    main()
