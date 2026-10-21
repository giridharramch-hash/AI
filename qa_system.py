"""
Extractive Question Answering on Course Material
using a Pretrained Transformer Encoder  (AI Mini-Capstone, topic 3.4)

Pipeline:   question
              -> TF-IDF retriever          (finds the top-k relevant passages)
              -> transformer encoder       (DistilBERT fine-tuned on SQuAD)
              -> QA head: start/end logits (one score per token, per boundary)
              -> best (start, end) span    (the answer is COPIED from the passage)

Run:
    python qa_system.py                       # interactive mode
    python qa_system.py -q "What is A* search?"
"""
import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# DistilBERT = 6-layer transformer ENCODER (66M params), distilled from BERT and
# fine-tuned on SQuAD v1.1. Small enough to run on a laptop CPU.
DEFAULT_MODEL = "distilbert-base-cased-distilled-squad"


# --------------------------------------------------------------------------- #
# 1. Data: loading and chunking the course material
# --------------------------------------------------------------------------- #
@dataclass
class Passage:
    pid: int
    unit: str
    title: str
    text: str


def split_sentences(text):
    """Very small rule-based sentence splitter (good enough for lecture notes)."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if s.strip()]


def chunk_text(text, max_words=120):
    """Split an over-long paragraph into sentence groups of <= max_words words.
    The reader sees at most ~384 tokens, so passages must stay short."""
    if len(text.split()) <= max_words:
        return [text]
    chunks, current = [], []
    for sent in split_sentences(text):
        if current and len(" ".join(current + [sent]).split()) > max_words:
            chunks.append(" ".join(current))
            current = [current[-1]]          # 1-sentence overlap keeps context
        current.append(sent)
    chunks.append(" ".join(current))
    return chunks


def load_passages(path):
    """Course material format: blocks separated by a blank line.
    The first line of a block may be a header like  [Search] A* search ."""
    raw = Path(path).read_text(encoding="utf-8")
    passages = []
    for block in (b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()):
        head, _, rest = block.partition("\n")
        m = re.match(r"\[(.+?)\]\s*(.*)", head)
        unit, title, body = (m.group(1), m.group(2), rest) if (m and rest) else ("General", "", block)
        for chunk in chunk_text(" ".join(body.split())):
            passages.append(Passage(len(passages), unit, title, chunk))
    return passages


# --------------------------------------------------------------------------- #
# 2. Retrieval (classical IR, no neural network)
# --------------------------------------------------------------------------- #
class TfidfRetriever:
    """Ranks passages by cosine similarity between TF-IDF vectors."""

    def __init__(self, passages):
        self.passages = passages
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([f"{p.title}. {p.text}" for p in passages])

    def search(self, question, k=3):
        sims = cosine_similarity(self.vectorizer.transform([question]), self.matrix).ravel()
        order = sims.argsort()[::-1][:k]
        return [(self.passages[i], float(sims[i])) for i in order]


# --------------------------------------------------------------------------- #
# 3. Baseline: return the single best-matching SENTENCE (no transformer)
# --------------------------------------------------------------------------- #
@dataclass
class Answer:
    text: str
    confidence: float
    passage: Passage = None
    retrieval_score: float = 0.0
    char_span: tuple = None
    candidates: list = field(default_factory=list)   # (passage, retrieval_score, answer_text, confidence)


class SentenceBaseline:
    def __init__(self, passages):
        self.items = [(p, s) for p in passages for s in split_sentences(p.text)]
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([s for _, s in self.items])

    def ask(self, question):
        sims = cosine_similarity(self.vectorizer.transform([question]), self.matrix).ravel()
        i = int(sims.argmax())
        passage, sentence = self.items[i]
        return Answer(text=sentence, confidence=float(sims[i]), passage=passage)


# --------------------------------------------------------------------------- #
# 4. Reader: pretrained transformer encoder + span-prediction head
# --------------------------------------------------------------------------- #
class TransformerReader:
    """
    Input to the encoder:  [CLS] question tokens [SEP] passage tokens [SEP]
    Output of the model:   for EVERY token, a start logit and an end logit
                           (a linear layer on top of the encoder's hidden states).
    Answer = the span (i, j) with i <= j that maximises P(start=i) * P(end=j),
             restricted to tokens that belong to the passage.
    """

    def __init__(self, model_name=DEFAULT_MODEL, max_answer_tokens=30, max_length=384):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device).eval()
        self.max_answer_tokens = max_answer_tokens
        self.max_length = max_length

    def _forward(self, question, context):
        torch = self.torch
        enc = self.tokenizer(
            question, context,
            return_tensors="pt", truncation="only_second",
            max_length=self.max_length, return_offsets_mapping=True,
        )
        offsets = enc["offset_mapping"][0].tolist()           # token -> (char_start, char_end)
        seq_ids = enc.sequence_ids(0)                         # None=special, 0=question, 1=passage
        inputs = {k: v.to(self.device) for k, v in enc.items() if k in self.tokenizer.model_input_names}
        with torch.no_grad():
            out = self.model(**inputs)
        # Only passage tokens may be answer boundaries: push all other logits to -inf
        is_context = torch.tensor([s == 1 for s in seq_ids], device=out.start_logits.device)
        start_logits = out.start_logits[0].masked_fill(~is_context, -1e9)
        end_logits = out.end_logits[0].masked_fill(~is_context, -1e9)
        start_p = torch.softmax(start_logits, dim=-1)          # probability distribution over tokens
        end_p = torch.softmax(end_logits, dim=-1)
        return enc, offsets, start_p, end_p

    def read(self, question, context):
        """Return (answer_text, confidence, (char_start, char_end))."""
        torch = self.torch
        _, offsets, start_p, end_p = self._forward(question, context)
        n = start_p.shape[0]
        # score[i, j] = P(start=i) * P(end=j); keep only 0 <= j - i < max_answer_tokens
        score = start_p[:, None] * end_p[None, :]
        ones = torch.ones(n, n, device=score.device)
        band = torch.triu(ones, diagonal=0) - torch.triu(ones, diagonal=self.max_answer_tokens)
        score = score * band
        i, j = divmod(int(torch.argmax(score)), n)
        c0, c1 = offsets[i][0], offsets[j][1]
        return context[c0:c1], float(score[i, j]), (c0, c1)

    def inspect(self, question, context, top_n=5):
        """For explanation/viva: the most likely start and end tokens."""
        enc, _, start_p, end_p = self._forward(question, context)
        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        top = lambda p: [(tokens[int(k)], round(float(p[k]), 4)) for k in p.topk(top_n).indices]
        return {"start": top(start_p), "end": top(end_p)}


# --------------------------------------------------------------------------- #
# 5. Full pipeline
# --------------------------------------------------------------------------- #
class ExtractiveQA:
    """retriever -> reader. Final answer = the span the reader is most confident about
    among the top-k retrieved passages."""

    def __init__(self, passages, reader, top_k=3):
        self.retriever = TfidfRetriever(passages)
        self.reader = reader
        self.top_k = top_k

    def ask(self, question):
        candidates = []
        for passage, r_score in self.retriever.search(question, self.top_k):
            text, conf, span = self.reader.read(question, passage.text)
            candidates.append((passage, r_score, text, conf, span))
        passage, r_score, text, conf, span = max(candidates, key=lambda c: c[3])
        return Answer(
            text=text, confidence=conf, passage=passage, retrieval_score=r_score, char_span=span,
            candidates=[(p, r, t, c) for p, r, t, c, _ in candidates],
        )


# --------------------------------------------------------------------------- #
# 6. Command-line demo
# --------------------------------------------------------------------------- #
def show(ans):
    where = f"[{ans.passage.unit}] {ans.passage.title}" if ans.passage.title else ans.passage.unit
    flag = "   (low confidence - check the source)" if ans.confidence < 0.2 else ""
    print(f"\n  Answer      : {ans.text}")
    print(f"  Confidence  : {ans.confidence:.3f}{flag}")
    print(f"  Source      : {where}")
    for p, r, t, c in ans.candidates:
        print(f"    - {p.title or p.unit:<42} retrieval={r:.3f}  span='{t}'  conf={c:.3f}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Extractive QA on course material")
    ap.add_argument("--material", default="course_material.txt")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("-q", "--question")
    args = ap.parse_args()

    passages = load_passages(args.material)
    print(f"Loaded {len(passages)} passages from {args.material}. Loading model '{args.model}' ...")
    qa = ExtractiveQA(passages, TransformerReader(args.model), top_k=args.top_k)

    if args.question:
        show(qa.ask(args.question))
        return
    print("Ask a question about the course material (empty line to quit).")
    while True:
        q = input("Q> ").strip()
        if not q:
            break
        show(qa.ask(q))


if __name__ == "__main__":
    main()
