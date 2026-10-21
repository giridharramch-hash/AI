import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("capstone_backend")

# Ensure workspace root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import existing ML code without modifying logic
import qa_system
import attention_demo

app = FastAPI(
    title="Extractive QA Course Material API",
    description="Backend API for AI Mini-Capstone 25CS301 (Topic 3.4 LLMs and Transformer Models)",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system state
passages: List[qa_system.Passage] = []
retriever: Optional[qa_system.TfidfRetriever] = None
baseline: Optional[qa_system.SentenceBaseline] = None
reader: Optional[qa_system.TransformerReader] = None
is_model_available = False
model_error_message = ""

def load_system():
    global passages, retriever, baseline, reader, is_model_available, model_error_message
    material_path = BASE_DIR / "course_material.txt"
    if not material_path.exists():
        logger.error(f"Course material file not found at {material_path}")
        return

    logger.info(f"Loading course material from {material_path}...")
    passages = qa_system.load_passages(str(material_path))
    retriever = qa_system.TfidfRetriever(passages)
    baseline = qa_system.SentenceBaseline(passages)
    logger.info(f"Loaded {len(passages)} passages into retriever and baseline.")

    try:
        logger.info(f"Loading transformer reader ({qa_system.DEFAULT_MODEL})...")
        reader = qa_system.TransformerReader(qa_system.DEFAULT_MODEL)
        is_model_available = True
        logger.info("DistilBERT Transformer Reader loaded successfully!")
    except Exception as e:
        is_model_available = False
        model_error_message = str(e)
        logger.warning(f"Transformer reader failed to load ({e}). Gracefully falling back to sentence baseline.")

# Call startup loader
load_system()

# Request schemas
class AskRequest(BaseModel):
    question: str
    top_k: int = 3

@app.get("/api/status")
def get_status():
    return {
        "status": "ready",
        "model_available": is_model_available,
        "model_name": qa_system.DEFAULT_MODEL if is_model_available else None,
        "device": getattr(reader, "device", "cpu") if is_model_available else "cpu",
        "passages_count": len(passages),
        "fallback_warning": model_error_message if not is_model_available else None
    }

@app.post("/api/ask")
def ask_question(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    if len(question) > 500:
        question = question[:500]

    top_k = max(1, min(req.top_k, 10))

    # Compute sentence baseline answer for side-by-side comparison
    base_ans = baseline.ask(question)
    base_char_start = base_ans.passage.text.find(base_ans.text)
    base_char_end = (base_char_start + len(base_ans.text)) if base_char_start != -1 else (0, len(base_ans.text))
    base_span = [base_char_start if base_char_start != -1 else 0, base_char_end]

    baseline_data = {
        "text": base_ans.text,
        "confidence": round(float(base_ans.confidence), 4),
        "char_span": base_span,
        "passage": {
            "id": base_ans.passage.pid,
            "unit": base_ans.passage.unit,
            "title": base_ans.passage.title,
            "text": base_ans.passage.text,
        },
        "source": {
            "id": base_ans.passage.pid,
            "unit": base_ans.passage.unit,
            "title": base_ans.passage.title,
            "label": f"[{base_ans.passage.unit}] {base_ans.passage.title}" if base_ans.passage.title else base_ans.passage.unit
        }
    }

    # If transformer model is not available, fall back cleanly
    if not is_model_available or reader is None:
        return {
            "status": "fallback",
            "is_fallback": True,
            "fallback_message": "Transformer model unavailable. Running sentence baseline fallback.",
            "answer": base_ans.text,
            "confidence": round(float(base_ans.confidence), 4),
            "char_span": base_span,
            "source": baseline_data["source"],
            "passage": baseline_data["passage"],
            "candidates": [{
                "passage": baseline_data["passage"],
                "retrieval_score": round(float(base_ans.confidence), 4),
                "span": base_ans.text,
                "confidence": round(float(base_ans.confidence), 4),
                "char_span": base_span,
            }],
            "baseline": baseline_data,
            "inspection": None
        }

    # Full extractive QA pipeline: retrieve top-k -> read each -> pick best
    try:
        retrieved_passages = retriever.search(question, top_k)
        candidates = []
        for p, r_score in retrieved_passages:
            ans_text, conf, span = reader.read(question, p.text)
            candidates.append({
                "passage": {
                    "id": p.pid,
                    "unit": p.unit,
                    "title": p.title,
                    "text": p.text
                },
                "retrieval_score": round(float(r_score), 4),
                "span": ans_text,
                "confidence": round(float(conf), 4),
                "char_span": [int(span[0]), int(span[1])]
            })

        # Best candidate by reader confidence
        best = max(candidates, key=lambda c: c["confidence"])

        # Detailed inspect: token logits for the winning passage
        try:
            inspection = reader.inspect(question, best["passage"]["text"], top_n=5)
        except Exception as e:
            logger.warning(f"Inspection failed: {e}")
            inspection = None

        return {
            "status": "success",
            "is_fallback": False,
            "answer": best["span"],
            "confidence": best["confidence"],
            "char_span": best["char_span"],
            "source": {
                "id": best["passage"]["id"],
                "unit": best["passage"]["unit"],
                "title": best["passage"]["title"],
                "label": f"[{best['passage']['unit']}] {best['passage']['title']}" if best['passage']['title'] else best['passage']['unit']
            },
            "passage": best["passage"],
            "candidates": candidates,
            "baseline": baseline_data,
            "inspection": inspection
        }
    except Exception as e:
        logger.error(f"Error during ask_question: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

@app.get("/api/passages")
def get_passages():
    return [
        {
            "id": p.pid,
            "unit": p.unit,
            "title": p.title,
            "text": p.text,
            "word_count": len(p.text.split())
        }
        for p in passages
    ]

@app.get("/api/results")
def get_results():
    results_path = BASE_DIR / "results.json"
    if not results_path.exists():
        return {
            "status": "not_generated",
            "message": "Results have not been generated yet. Please run python evaluate.py."
        }
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
        data["status"] = "success"
        return data
    except Exception as e:
        logger.error(f"Failed to read results.json: {e}")
        return {
            "status": "error",
            "message": f"Failed to parse results.json: {str(e)}"
        }

@app.get("/api/attention")
def get_attention(d_k: float = Query(4.0, ge=1.0, le=256.0), scale: bool = Query(True)):
    """
    Computes toy self-attention weights and returns token attention matrix.
    Reuses attention_demo.self_attention with seed 0.
    """
    rng = np.random.default_rng(0)
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    d_model = 8
    d_k_int = int(d_k) if d_k > 0 else 4
    
    # Toy embeddings and projection weights
    X = rng.normal(size=(len(tokens), d_model))
    Wq = rng.normal(size=(d_model, d_k_int))
    Wk = rng.normal(size=(d_model, d_k_int))
    Wv = rng.normal(size=(d_model, d_k_int))
    
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores = Q @ K.T
    if scale:
        scores = scores / np.sqrt(d_k)
    weights = attention_demo.softmax(scores)
    
    # Calculate row sums and max weight distribution
    row_sums = [round(float(s), 4) for s in weights.sum(axis=1)]
    weights_matrix = [[round(float(v), 4) for v in row] for row in weights]
    mean_max_weight = round(float(weights.max(axis=1).mean()), 4)

    return {
        "tokens": tokens,
        "weights": weights_matrix,
        "row_sums": row_sums,
        "d_k": d_k,
        "scale": scale,
        "mean_max_weight": mean_max_weight,
        "formula": "softmax(QK^T / sqrt(d_k))" if scale else "softmax(QK^T)"
    }

@app.get("/api/poster")
def download_poster():
    poster_path = BASE_DIR / "poster.pdf"
    if not poster_path.exists():
        # Try generating poster
        try:
            import subprocess
            subprocess.run([sys.executable, str(BASE_DIR / "make_poster.py")], cwd=str(BASE_DIR), check=True)
        except Exception as e:
            logger.warning(f"Could not generate poster on the fly: {e}")

    if poster_path.exists():
        return FileResponse(
            str(poster_path),
            media_type="application/pdf",
            filename="AI_Capstone_Technical_Poster.pdf"
        )
    raise HTTPException(status_code=404, detail="poster.pdf not found and could not be generated.")

# Mount frontend directory for full-stack delivery
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
