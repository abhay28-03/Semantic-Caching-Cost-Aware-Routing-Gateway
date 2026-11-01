import time
import requests

API_URL = "http://localhost:8080/v1/chat/completions"
INVALIDATE_URL = "http://localhost:8080/v1/cache/invalidate"

test_prompts = [
    # Tier 1: Factual / Lightweight (Should pick gpt-oss-20b)
    "What is the chemical symbol for Gold?",
    "Tell me the chemical symbol representing gold.",
    
    # Tier 2: Analytical / Heavy (Routed to heavy model)
    "How does Paxos handle network partitions differently than Raft consensus?",
    
    # Tier 3: Definitions (Lightweight model)
    "What is an HTTP 502 Bad Gateway status?",
    
    # Tier 4: Architectural Complexity (Routed to heavy model)
    "What are the latency bottlenecks when querying 10 million vector records in HNSW graph indexes?"
]

print("=== Phase 1: Running Semantic Routing Benchmark ===")
for q in test_prompts:
    try:
        response = requests.post(API_URL, json={"query": q})
        if response.status_code != 200:
            print(f"[ERROR {response.status_code}] Failed for '{q[:30]}...': {response.text}")
            continue
        
        res = response.json()
        r_score = f"Align: {res['router_score']}" if res.get('router_score') is not None else "Cached"
        print(f"[{res['source'].upper():<8}] {res['latency_ms']:>6.2f}ms | Model: {res['model_used']:<24} | {r_score:<13} | Q: {q[:30]}")
    except Exception as e:
        print(f"[REQUEST FAILED] {e}")
    time.sleep(0.5)

print("\n=== Phase 2: Testing Cache Invalidation Endpoint ===")
try:
    inv_res = requests.post(INVALIDATE_URL, json={"query": "What is the chemical symbol for Gold?"}).json()
    print(f"Invalidation response: {inv_res}")

    retest_res = requests.post(API_URL, json={"query": "What is the chemical symbol for Gold?"}).json()
    print(f"After Invalidation: Source is now [{retest_res['source'].upper()}] (Latency: {retest_res['latency_ms']}ms)")
except Exception as e:
    print(f"[PHASE 2 ERROR] {e}")