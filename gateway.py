import os
import json
import time
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CACHE_INDEX_NAME = "idx:semantic_cache"
CACHE_PREFIX = "cache:"
EMBEDDING_DIM = 384
SIMILARITY_THRESHOLD = 0.88

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in .env")

app = FastAPI(title="Semantic Caching & Cost-Aware Router Gateway", version="2.0.0")

# Redis connection
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)

# Local Embedding Model
print("Loading local sentence-transformer model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Groq Client via OpenAI SDK
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

# ---------------------------------------------------------
# Commercial Pricing Baselines ($ / 1M tokens)
# ---------------------------------------------------------
MODEL_PRICING = {
    "small": {
        "id": "openai/gpt-oss-20b",
        "input_cost_per_m": 0.15,
        "output_cost_per_m": 0.60,
    },
    "heavy": {
        "id": "openai/gpt-oss-120b",
        "input_cost_per_m": 1.20,
        "output_cost_per_m": 4.50,
    },
}

# ---------------------------------------------------------
# Semantic Router Head Anchors
# ---------------------------------------------------------
# We build reference centroid anchors for "Reasoning / Heavy" tasks.
REASONING_ANCHORS = [
    "Compare and contrast the trade-offs between architectural designs.",
    "Explain step-by-step mathematical proof and algorithmic convergence.",
    "Debug complex distributed systems deadlock and concurrency failure.",
    "Formulate an optimization plan for high-scale database partitioning.",
    "Analyze security vulnerabilities and edge cases in deep detail.",
    "Provide an in-depth breakdown of pros, cons, and performance implications."
]

print("Precomputing reasoning centroid vectors...")
reasoning_anchor_vectors = embed_model.encode(REASONING_ANCHORS, normalize_embeddings=True)
REASONING_CENTROID = np.mean(reasoning_anchor_vectors, axis=0)
REASONING_CENTROID = REASONING_CENTROID / np.linalg.norm(REASONING_CENTROID)

# ---------------------------------------------------------
# Initialize Redis Vector Index
# ---------------------------------------------------------
def init_redis_index():
    try:
        redis_client.ft(CACHE_INDEX_NAME).info()
    except Exception:
        schema = (
            TextField("query_text"),
            TextField("response_text"),
            TextField("routed_model"),
            VectorField(
                "vector",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": EMBEDDING_DIM,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
        )
        redis_client.ft(CACHE_INDEX_NAME).create_index(
            schema,
            definition=IndexDefinition(prefix=[CACHE_PREFIX], index_type=IndexType.HASH),
        )

init_redis_index()

# ---------------------------------------------------------
# Core Helper Functions
# ---------------------------------------------------------
def get_local_embedding(text: str) -> Tuple[np.ndarray, bytes]:
    """Generates normalized float32 numpy vector and raw byte array."""
    vec = embed_model.encode(text, normalize_embeddings=True).astype("float32")
    return vec, vec.tobytes()

def decode_val(val: Any) -> str:
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val) if val is not None else ""

def classify_query_complexity(query: str, query_vec: np.ndarray) -> Tuple[str, str, float]:
    """
    Semantic Router: Measures Cosine Similarity between prompt vector
    and precomputed Reasoning Centroid vector.
    """
    # Cosine similarity between normalized vectors = dot product
    sim_to_reasoning = float(np.dot(query_vec, REASONING_CENTROID))
    
    # Change len > 35 to > 25, and change 0.52 to 0.25:
    if len(query.split()) > 25 or sim_to_reasoning >= 0.25:
        return "heavy", MODEL_PRICING["heavy"]["id"], round(sim_to_reasoning, 4)
    
    return "small", MODEL_PRICING["small"]["id"], round(sim_to_reasoning, 4)

def check_semantic_cache(query_vector_bytes: bytes) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    q = (
        Query("*=>[KNN 1 @vector $vec AS score]")
        .sort_by("score")
        .return_fields("query_text", "response_text", "routed_model", "score")
        .dialect(2)
    )
    
    try:
        results = redis_client.ft(CACHE_INDEX_NAME).search(
            q, query_params={"vec": query_vector_bytes}
        )
    except Exception as e:
        print(f"[CACHE SEARCH ERROR] {e}")
        return None, None

    if results and results.docs:
        best_match = results.docs[0]
        cosine_distance = float(best_match.score)
        similarity = 1.0 - cosine_distance

        if similarity >= SIMILARITY_THRESHOLD:
            return {
                "key": best_match.id,
                "matched_query": decode_val(getattr(best_match, "query_text", "")),
                "response_text": decode_val(getattr(best_match, "response_text", "")),
                "routed_model": decode_val(getattr(best_match, "routed_model", MODEL_PRICING["small"]["id"])),
            }, round(similarity, 4)
            
    return None, None

def store_in_cache(query: str, response: str, model_used: str, query_vector_bytes: bytes, ttl: int = 86400):
    key = f"{CACHE_PREFIX}{abs(hash(query))}"
    mapping = {
        "query_text": query,
        "response_text": response,
        "routed_model": model_used,
        "vector": query_vector_bytes,
    }
    redis_client.hset(key, mapping=mapping)
    redis_client.expire(key, ttl)

# ---------------------------------------------------------
# Decoupled Asynchronous Telemetry
# ---------------------------------------------------------
def async_log_telemetry_event(
    query: str,
    source: str,
    model_used: str,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    similarity: Optional[float] = None,
    router_score: Optional[float] = None
):
    try:
        baseline_cost = (input_tokens / 1_000_000) * MODEL_PRICING["heavy"]["input_cost_per_m"] + \
                        (output_tokens / 1_000_000) * MODEL_PRICING["heavy"]["output_cost_per_m"]
        
        tier = "small" if model_used == MODEL_PRICING["small"]["id"] else "heavy"
        actual_cost = 0.0
        if source != "cache":
            actual_cost = (input_tokens / 1_000_000) * MODEL_PRICING[tier]["input_cost_per_m"] + \
                          (output_tokens / 1_000_000) * MODEL_PRICING[tier]["output_cost_per_m"]
            
        savings_usd = max(0.0, baseline_cost - actual_cost)

        event = {
            "timestamp": time.time(),
            "query": query[:65],
            "source": source,
            "model_used": model_used,
            "latency_ms": round(latency_ms, 2),
            "actual_cost_usd": round(actual_cost, 6),
            "savings_usd": round(savings_usd, 6),
            "similarity": round(float(similarity), 4) if similarity is not None else 0.0,
            "router_score": router_score
        }

        redis_client.lpush("telemetry:requests", json.dumps(event))
        redis_client.ltrim("telemetry:requests", 0, 500)
    except Exception as e:
        print(f"[ASYNC TELEMETRY ERROR] {e}")

# ---------------------------------------------------------
# Schemas
# ---------------------------------------------------------
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    source: str
    model_used: str
    latency_ms: float
    cost_usd: float
    similarity: Optional[float] = None
    router_score: Optional[float] = None

class InvalidateRequest(BaseModel):
    query: Optional[str] = None
    purge_all: bool = False

# ---------------------------------------------------------
# Application Endpoints
# ---------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "online",
        "engine": "v2-semantic-router",
        "dashboard": "http://localhost:8080/dashboard",
        "docs": "http://localhost:8080/docs"
    }

@app.post("/v1/chat/completions", response_model=QueryResponse)
def execute_query(req: QueryRequest, background_tasks: BackgroundTasks):
    start_time = time.time()
    
    # 1. Compute dense embedding vector
    query_vec_np, query_vec_bytes = get_local_embedding(req.query)
    
    # 2. Check semantic cache
    cached_result, similarity = check_semantic_cache(query_vec_bytes)
    
    if cached_result:
        latency_ms = (time.time() - start_time) * 1000
        answer = cached_result["response_text"]
        model_name = cached_result["routed_model"]

        background_tasks.add_task(
            async_log_telemetry_event,
            query=req.query,
            source="cache",
            model_used=model_name,
            latency_ms=latency_ms,
            input_tokens=len(req.query.split()) * 2,
            output_tokens=len(answer.split()) * 2,
            similarity=similarity,
            router_score=None
        )
        
        return QueryResponse(
            answer=answer,
            source="cache",
            model_used=model_name,
            latency_ms=round(latency_ms, 2),
            cost_usd=0.0,
            similarity=similarity,
            router_score=None
        )

    # 3. Cache Miss: Semantic Routing Head
    tier, selected_model, router_score = classify_query_complexity(req.query, query_vec_np)
    
    try:
        completion = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": "You are a concise, helpful assistant."},
                {"role": "user", "content": req.query}
            ],
            temperature=0.2,
        )
    except Exception as err:
        print(f"[GROQ PRIMARY FAILED] {selected_model}: {err}. Retrying with small model...")
        selected_model = MODEL_PRICING["small"]["id"]
        tier = "small"
        completion = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": "You are a concise, helpful assistant."},
                {"role": "user", "content": req.query}
            ],
            temperature=0.2,
        )

    answer_text = completion.choices[0].message.content or ""
    latency_ms = (time.time() - start_time) * 1000
    
    usage = getattr(completion, "usage", None)
    input_tokens = usage.prompt_tokens if usage and getattr(usage, "prompt_tokens", None) else len(req.query.split()) * 2
    output_tokens = usage.completion_tokens if usage and getattr(usage, "completion_tokens", None) else len(answer_text.split()) * 2

    # 4. Asynchronous Cache & Telemetry Persistence
    background_tasks.add_task(store_in_cache, req.query, answer_text, selected_model, query_vec_bytes)
    background_tasks.add_task(
        async_log_telemetry_event,
        query=req.query,
        source="groq_api",
        model_used=selected_model,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        similarity=None,
        router_score=router_score
    )
    
    actual_cost = (
        (input_tokens / 1_000_000) * MODEL_PRICING[tier]["input_cost_per_m"] +
        (output_tokens / 1_000_000) * MODEL_PRICING[tier]["output_cost_per_m"]
    )

    return QueryResponse(
        answer=answer_text,
        source="groq_api",
        model_used=selected_model,
        latency_ms=round(latency_ms, 2),
        cost_usd=round(actual_cost, 6),
        similarity=None,
        router_score=router_score
    )

@app.post("/v1/cache/invalidate")
def invalidate_cache(req: InvalidateRequest):
    """
    Purges either the entire cache or all semantically matched entries.
    """
    if req.purge_all:
        keys = redis_client.keys(f"{CACHE_PREFIX}*")
        deleted_count = 0
        if keys:
            deleted_count = redis_client.delete(*keys)
        return {"status": "success", "message": f"Purged {deleted_count} cache entries."}
    
    if not req.query:
        raise HTTPException(status_code=400, detail="Must provide either 'query' or 'purge_all: true'")
    
    _, vec_bytes = get_local_embedding(req.query)
    
    # KNN query looking for top 5 close semantic duplicates to purge
    q = (
        Query("*=>[KNN 5 @vector $vec AS score]")
        .sort_by("score")
        .return_fields("query_text", "score")
        .dialect(2)
    )
    
    results = redis_client.ft(CACHE_INDEX_NAME).search(q, query_params={"vec": vec_bytes})
    deleted_keys = []
    
    if results and results.docs:
        for doc in results.docs:
            sim = 1.0 - float(doc.score)
            if sim >= SIMILARITY_THRESHOLD:
                redis_client.delete(doc.id)
                deleted_keys.append(doc.id)
                
    if deleted_keys:
        return {
            "status": "success",
            "purged_keys": deleted_keys,
            "count": len(deleted_keys)
        }
    
    return {"status": "not_found", "message": "No matching semantic entry found to invalidate."}

@app.get("/telemetry/data")
def get_telemetry_data():
    raw_events = redis_client.lrange("telemetry:requests", 0, 100)
    events = []
    if raw_events:
        for e in raw_events:
            text = e.decode("utf-8") if isinstance(e, bytes) else e
            events.append(json.loads(text))
    
    total = len(events)
    hits = sum(1 for e in events if e.get("source") == "cache")
    hit_rate = round((hits / total * 100), 1) if total > 0 else 0.0
    
    total_savings = sum(e.get("savings_usd", 0.0) for e in events)
    avg_latency = round(sum(e.get("latency_ms", 0.0) for e in events) / total, 2) if total > 0 else 0.0

    return {
        "summary": {
            "total_requests": total,
            "cache_hit_rate": hit_rate,
            "total_savings_usd": round(total_savings, 5),
            "avg_latency_ms": avg_latency
        },
        "recent_events": events[:25]
    }

@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Semantic Cache & Router Gateway v2</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body class="bg-slate-900 text-slate-100 p-8 font-sans">
        <div class="max-w-6xl mx-auto space-y-6">
            <header class="flex justify-between items-center border-b border-slate-700 pb-4">
                <div>
                    <h1 class="text-2xl font-bold text-white flex items-center gap-2">
                        Semantic Cache & Gateway Telemetry
                        <span class="text-xs font-mono bg-indigo-900/80 text-indigo-300 px-2 py-0.5 rounded border border-indigo-700/50">v2.0-async</span>
                    </h1>
                    <p class="text-sm text-slate-400">Embedding-based routing, decoupled async telemetry & cache control</p>
                </div>
                <div class="flex gap-2">
                    <button onclick="purgeCache()" class="bg-rose-700 hover:bg-rose-600 text-xs px-3 py-2 rounded font-medium transition">Purge Cache</button>
                    <button onclick="fetchStats()" class="bg-indigo-600 hover:bg-indigo-500 text-xs px-3 py-2 rounded font-medium transition">Refresh</button>
                </div>
            </header>

            <div class="grid grid-cols-4 gap-4">
                <div class="bg-slate-800 p-5 rounded-lg border border-slate-700">
                    <div class="text-xs text-slate-400 uppercase tracking-wide">Total Requests</div>
                    <div id="stat-total" class="text-3xl font-extrabold mt-1 text-white">0</div>
                </div>
                <div class="bg-slate-800 p-5 rounded-lg border border-slate-700">
                    <div class="text-xs text-slate-400 uppercase tracking-wide">Cache Hit Rate</div>
                    <div id="stat-hitrate" class="text-3xl font-extrabold mt-1 text-emerald-400">0%</div>
                </div>
                <div class="bg-slate-800 p-5 rounded-lg border border-slate-700">
                    <div class="text-xs text-slate-400 uppercase tracking-wide">Avg Latency</div>
                    <div id="stat-latency" class="text-3xl font-extrabold mt-1 text-sky-400">0 ms</div>
                </div>
                <div class="bg-slate-800 p-5 rounded-lg border border-slate-700">
                    <div class="text-xs text-slate-400 uppercase tracking-wide">Simulated Savings</div>
                    <div id="stat-savings" class="text-3xl font-extrabold mt-1 text-amber-400">$0.00</div>
                </div>
            </div>

            <div class="grid grid-cols-3 gap-4">
                <div class="bg-slate-800 p-4 rounded-lg border border-slate-700 col-span-2">
                    <h3 class="text-xs uppercase font-semibold text-slate-400 mb-2">Request Latencies (Recent 20)</h3>
                    <div class="h-48">
                        <canvas id="latencyChart"></canvas>
                    </div>
                </div>
                <div class="bg-slate-800 p-4 rounded-lg border border-slate-700">
                    <h3 class="text-xs uppercase font-semibold text-slate-400 mb-2">Traffic Distribution</h3>
                    <div class="h-48 flex items-center justify-center">
                        <canvas id="ratioChart"></canvas>
                    </div>
                </div>
            </div>

            <div class="bg-slate-800 rounded-lg border border-slate-700 overflow-hidden shadow-sm">
                <div class="p-4 border-b border-slate-700 font-semibold text-sm">Recent Traffic Ingestion</div>
                <table class="w-full text-left text-xs">
                    <thead class="bg-slate-800/80 text-slate-400 uppercase border-b border-slate-700">
                        <tr>
                            <th class="p-3">Query</th>
                            <th class="p-3">Source</th>
                            <th class="p-3">Model</th>
                            <th class="p-3">Semantic Match</th>
                            <th class="p-3">Router Alignment</th>
                            <th class="p-3">Latency</th>
                            <th class="p-3">Cost</th>
                        </tr>
                    </thead>
                    <tbody id="events-table" class="divide-y divide-slate-700/50"></tbody>
                </table>
            </div>
        </div>

        <script>
            let latencyChart = null;
            let ratioChart = null;

            function initCharts() {
                const ctxLat = document.getElementById('latencyChart').getContext('2d');
                latencyChart = new Chart(ctxLat, {
                    type: 'bar',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Latency (ms)',
                            data: [],
                            backgroundColor: [],
                            borderRadius: 4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { display: false },
                            y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } }
                        }
                    }
                });

                const ctxRatio = document.getElementById('ratioChart').getContext('2d');
                ratioChart = new Chart(ctxRatio, {
                    type: 'doughnut',
                    data: {
                        labels: ['Cache Hits', 'API Misses'],
                        datasets: [{
                            data: [0, 0],
                            backgroundColor: ['#10b981', '#6366f1'],
                            borderWidth: 0
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { position: 'bottom', labels: { color: '#cbd5e1', font: { size: 10 } } }
                        }
                    }
                });
            }

            async function purgeCache() {
                if(!confirm("Purge all semantic cached keys?")) return;
                await fetch('/v1/cache/invalidate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({purge_all: true})
                });
                fetchStats();
            }

            async function fetchStats() {
                try {
                    const res = await fetch('/telemetry/data');
                    const data = await res.json();
                    
                    document.getElementById('stat-total').innerText = data.summary.total_requests;
                    document.getElementById('stat-hitrate').innerText = data.summary.cache_hit_rate + '%';
                    document.getElementById('stat-latency').innerText = data.summary.avg_latency_ms + ' ms';
                    document.getElementById('stat-savings').innerText = '$' + Number(data.summary.total_savings_usd).toFixed(4);

                    const tbody = document.getElementById('events-table');
                    tbody.innerHTML = '';
                    
                    const reversedEvents = [...data.recent_events].reverse();
                    const latLabels = [];
                    const latData = [];
                    const latColors = [];
                    let cacheHits = 0;
                    let apiMisses = 0;

                    reversedEvents.forEach((e, idx) => {
                        const isHit = e.source === 'cache';
                        if (isHit) cacheHits++; else apiMisses++;
                        
                        latLabels.push(`Q${idx + 1}`);
                        latData.push(e.latency_ms);
                        latColors.push(isHit ? '#10b981' : '#6366f1');

                        const row = document.createElement('tr');
                        const badgeClass = isHit ? 'bg-emerald-900/60 text-emerald-300 border border-emerald-700/50' : 'bg-indigo-900/60 text-indigo-300 border border-indigo-700/50';
                        row.innerHTML = `
                            <td class="p-3 font-mono">${e.query}...</td>
                            <td class="p-3"><span class="px-2 py-0.5 rounded text-[10px] font-semibold ${badgeClass}">${e.source.toUpperCase()}</span></td>
                            <td class="p-3 text-slate-300 font-mono">${e.model_used}</td>
                            <td class="p-3 font-mono">${e.similarity ? Number(e.similarity).toFixed(3) : '-'}</td>
                            <td class="p-3 font-mono">${e.router_score !== null && e.router_score !== undefined ? Number(e.router_score).toFixed(3) : '-'}</td>
                            <td class="p-3 font-mono">${Number(e.latency_ms).toFixed(1)} ms</td>
                            <td class="p-3 font-mono">$${Number(e.actual_cost_usd).toFixed(6)}</td>
                        `;
                        tbody.prepend(row);
                    });

                    if (latencyChart && ratioChart) {
                        latencyChart.data.labels = latLabels;
                        latencyChart.data.datasets[0].data = latData;
                        latencyChart.data.datasets[0].backgroundColor = latColors;
                        latencyChart.update();

                        ratioChart.data.datasets[0].data = [cacheHits, apiMisses];
                        ratioChart.update();
                    }
                } catch (err) {
                    console.error('Failed to update telemetry:', err);
                }
            }

            initCharts();
            fetchStats();
            setInterval(fetchStats, 3000);
        </script>
    </body>
    </html>
    """