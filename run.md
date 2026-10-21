# Running the Extractive QA Capstone Website

This project provides a modern, fast, full-stack website for the AI Mini-Capstone project:  
**"Extractive Question Answering on Course Material using a Pretrained Transformer Encoder"**  
(Artificial Intelligence 25CS301, Topic 3.4 LLMs and Transformer Models).

---

## ⚡ Quick Start (One Command)

### 1. Activate the Virtual Environment
Windows PowerShell:
```powershell
.\.venv\Scripts\activate
```

*(If setting up on a new machine from scratch, simply run: `pip install -r requirements.txt`)*

### 2. Run the Full Application
```powershell
python run.py
```
Or directly with Uvicorn:
```powershell
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open your browser and navigate to:  
👉 **http://localhost:8000**

The FastAPI backend automatically serves both the JSON API endpoints and the frontend static single-page application at root `/`.

---

## 🛠️ System Architecture

- **Backend**: FastAPI (`/backend/app.py`), directly reusing `qa_system.py`:
  - `POST /api/ask`: Evaluates questions using `TfidfRetriever` + `TransformerReader` (DistilBERT), returning predicted span, confidence, char span, top-k candidate passages, and sentence-baseline comparison.
  - `GET /api/passages`: Returns all 21 course lecture passages.
  - `GET /api/results`: Reads empirical test results from `results.json`.
  - `GET /api/attention`: Computes interactive self-attention weights and row sums for token queries using `attention_demo.py`.
  - `GET /api/poster`: Serves or generates the A5 technical poster (`poster.pdf`).
  - `GET /api/status`: Reports model availability and device.
- **Frontend**: Lightweight, high-performance web interface:
  - Single-page layout with smooth sticky top navigation.
  - Interactive Live Demo with 6 instant-example chips and side-by-side baseline comparison.
  - KaTeX-rendered mathematical self-attention formulas and live attention heatmap.
  - Chart.js interactive grouped bar chart, top-k ablation, and per-unit breakdown.
  - Filterable error analysis table (diagnosing retrieval misses vs reader errors).
  - Searchable course material browser for all 21 passages.
  - Light / Dark mode toggle with persistent state.
  - Configurable team metadata in `frontend/js/config.js`.

---

## 🔬 Running Original CLI Scripts

All original command-line tools continue to work independently:

1. **Interactive Terminal QA Demo**:
   ```powershell
   python qa_system.py -q "Which company released BERT?"
   ```

2. **Evaluate Benchmark & Generate `results.json`**:
   ```powershell
   python evaluate.py
   ```

3. **Self-Attention Matplotlib Demo**:
   ```powershell
   python attention_demo.py
   ```

4. **Generate Technical Poster (A5 PDF)**:
   ```powershell
   python make_poster.py --team "Giridharram, Member 2, Member 3, Member 4"
   ```

---

## ⚙️ Configuration

To modify team names or demo links, edit `frontend/js/config.js`:
```javascript
window.CAPSTONE_CONFIG = {
  teamMembers: [
    "Your Name",
    "Team Member 2",
    "Team Member 3",
    "Team Member 4"
  ]
};
```
