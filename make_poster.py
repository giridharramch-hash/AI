"""
Generate the A5 technical poster (poster.pdf) from your REAL evaluation results.

    python evaluate.py                                   # 1) produces results.json
    python make_poster.py --team "Name1, Name2, Name3, Name4" [--qr https://github.com/you/repo]

Requires:  pip install reportlab
All numbers on the poster come from results.json - nothing is hard-coded.
"""
import argparse
import io
import json
import sys
from xml.sax.saxutils import escape

from reportlab.graphics import renderPDF
from reportlab.graphics.barcode import qr
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import A5
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

W, H = A5                       # 419.5 x 595.3 pt  (148 x 210 mm)
M, GAP = 13, 5
NAVY, BLUE = colors.HexColor("#14284b"), colors.HexColor("#2b6cb0")
GREY, LIGHT, INK = colors.HexColor("#9aa5b1"), colors.HexColor("#eef2f7"), colors.HexColor("#1f2933")


def pct(v):
    return f"{v * 100:.0f}%"


def para(html, size, bold=False, color=INK, leading=None):
    style = ParagraphStyle("p", fontName="Helvetica-Bold" if bold else "Helvetica", fontSize=size,
                           leading=leading or size * 1.27, textColor=color)
    return Paragraph(html, style)


def text_box(c, x, ytop, w, title, html, size):
    p = para(html, size)
    _, h = p.wrap(w - 10, 1000)
    total = 15 + h + 5
    c.setFillColor(LIGHT)
    c.roundRect(x, ytop - total, w, total, 4, stroke=0, fill=1)
    c.setFillColor(BLUE)
    c.setFont("Helvetica-Bold", size + 0.9)
    c.drawString(x + 5, ytop - 10.5, title.upper())
    p.drawOn(c, x + 5, ytop - total + 5)
    return ytop - total - GAP


def workflow_box(c, x, ytop, w, steps, size):
    bh, gap = 14, 6.5
    total = 15 + len(steps) * bh + (len(steps) - 1) * gap + 6
    c.setFillColor(LIGHT)
    c.roundRect(x, ytop - total, w, total, 4, stroke=0, fill=1)
    c.setFillColor(BLUE)
    c.setFont("Helvetica-Bold", size + 0.9)
    c.drawString(x + 5, ytop - 10.5, "METHODOLOGY / WORKFLOW")
    y = ytop - 15
    for i, s in enumerate(steps):
        c.setFillColor(NAVY if i in (2, 3) else colors.white)
        c.setStrokeColor(NAVY)
        c.roundRect(x + 8, y - bh, w - 16, bh, 3, stroke=1, fill=1)
        c.setFillColor(colors.white if i in (2, 3) else INK)
        c.setFont("Helvetica-Bold" if i in (2, 3) else "Helvetica", size)
        c.drawCentredString(x + w / 2, y - bh / 2 - size * 0.33, s)
        y -= bh
        if i < len(steps) - 1:
            c.setFillColor(BLUE)
            ax = x + w / 2
            p = c.beginPath()
            p.moveTo(ax - 3, y - 1.2)
            p.lineTo(ax + 3, y - 1.2)
            p.lineTo(ax, y - gap + 1)
            p.close()
            c.drawPath(p, stroke=0, fill=1)
            y -= gap
    return ytop - total - GAP


def results_box(c, x, ytop, w, R, base, main, size):
    metrics = [("EM", "EM"), ("F1", "F1"), ("Contains", "Contains")]
    lab_w, val_w, bar_h = 38, 22, 7.0
    chart_h = len(metrics) * (2 * (bar_h + 0.8) + 5) + 11
    info = para(
        f"Retriever Hit@{R['top_k']}: <b>{pct(R['retrieval_hit_at_k'][str(R['top_k'])])}</b> &nbsp;|&nbsp; "
        f"Latency: <b>{main['latency_ms']:.0f} ms</b>/question &nbsp;|&nbsp; "
        f"{R['n_questions']} test questions", size)
    _, ih = info.wrap(w - 10, 1000)
    total = 15 + chart_h + ih + 6
    c.setFillColor(LIGHT)
    c.roundRect(x, ytop - total, w, total, 4, stroke=0, fill=1)
    c.setFillColor(BLUE)
    c.setFont("Helvetica-Bold", size + 0.9)
    c.drawString(x + 5, ytop - 10.5, "KEY RESULTS")
    bar_max = w - 10 - lab_w - val_w
    y = ytop - 17
    for key, label in metrics:
        c.setFillColor(INK)
        c.setFont("Helvetica", size)
        c.drawString(x + 5, y - 10.0, label)
        for val, col in ((base[key], GREY), (main[key], BLUE)):
            c.setFillColor(col)
            c.rect(x + 5 + lab_w, y - bar_h, max(1.0, val * bar_max), bar_h, stroke=0, fill=1)
            c.setFillColor(INK)
            c.setFont("Helvetica-Bold", size - 1.3)
            c.drawString(x + 5 + lab_w + max(1.0, val * bar_max) + 2.5, y - bar_h + 1.6, pct(val))
            y -= bar_h + 0.8
        y -= 5
    # legend
    c.setFillColor(GREY)
    c.rect(x + 5, y - 3.5, 6, 4.5, stroke=0, fill=1)
    c.setFillColor(INK)
    c.setFont("Helvetica", size - 0.3)
    c.drawString(x + 13, y - 3, "TF-IDF sentence baseline")
    c.setFillColor(BLUE)
    c.rect(x + w / 2 + 2, y - 3.5, 6, 4.5, stroke=0, fill=1)
    c.setFillColor(INK)
    c.drawString(x + w / 2 + 10, y - 3, "Transformer reader")
    info.drawOn(c, x + 5, ytop - total + 4)
    return ytop - total - GAP


def draw(c, size, args, R):
    base = R["systems"]["baseline_sentence_tfidf"]
    main = R["systems"][R["main_system"]]
    k = R["top_k"]
    colw = (W - 2 * M - GAP) / 2

    # ---- header ----
    title = para(escape(args.title), size + 6.2, bold=True, color=colors.white)
    sub = para("AI Mini-Capstone &nbsp;|&nbsp; Artificial Intelligence (25CS301) &nbsp;|&nbsp; Topic 3.4 LLMs and Transformer Models",
               size - 0.2, color=colors.HexColor("#c7d6ee"))
    team = para("<b>Team:</b> " + escape(args.team), size, color=colors.white)
    hs = [p.wrap(W - 2 * M, 1000)[1] for p in (title, sub, team)]
    head_h = 10 + sum(hs) + 3 * 3 + 5
    c.setFillColor(NAVY)
    c.rect(0, H - head_h, W, head_h, stroke=0, fill=1)
    y = H - 9
    for p, h in zip((title, sub, team), hs):
        p.drawOn(c, M, y - h)
        y -= h + 3
    top = H - head_h - GAP + 1

    # ---- left column ----
    yl = top
    yl = text_box(c, M, yl, colw, "Problem Statement",
                  "Finding a short fact (e.g. <i>\"Which property makes A* optimal?\"</i>) in long lecture notes is slow. "
                  "Keyword search returns whole pages or sentences, not the answer. "
                  "<b>Goal:</b> given a question, return the exact answer span from the course material, "
                  "with its source and a confidence score.", size)
    yl = text_box(c, M, yl, colw, "Objectives",
                  "1. Build an <b>extractive QA</b> system over AI course notes.<br/>"
                  "2. Use a <b>pretrained transformer encoder</b> (DistilBERT) to predict answer start/end positions.<br/>"
                  f"3. Compare it with a TF-IDF sentence-retrieval baseline using EM, F1 and Hit@k.", size)
    yl = text_box(c, M, yl, colw, "AI Technique / Algorithm",
                  "<b>Reader:</b> transformer encoder (self-attention, 6 layers, 12 heads) fine-tuned on SQuAD. "
                  "Input <font face='Courier'>[CLS] question [SEP] passage [SEP]</font>. A linear QA head outputs a "
                  "<b>start logit and an end logit per token</b>; softmax gives probabilities; the answer is the span "
                  "maximising P(start) x P(end) inside the passage.<br/>"
                  f"<b>Retriever:</b> TF-IDF + cosine similarity selects the top-{k} passages; the most confident span wins.", size)
    yl = text_box(c, M, yl, colw, "Implementation",
                  f"Python, PyTorch, Hugging Face Transformers, scikit-learn. Corpus: <b>{R['n_passages']} passages</b> of AI course notes "
                  f"(search, knowledge-based agents, planning, LLMs); <b>{R['n_questions']} hand-written test questions</b> with gold answers. "
                  "Command-line demo prints the answer, confidence and source passage.", size)

    # ---- right column ----
    xr = M + colw + GAP
    yr = top
    yr = workflow_box(c, xr, yr, colw, ["Question", f"TF-IDF retriever (top-{k} passages)", "Transformer encoder (DistilBERT)",
                                        "QA head: start / end logits", "Best span = answer + confidence"], size - 0.4)
    yr = results_box(c, xr, yr, colw, R, base, main, size)
    yr = text_box(c, xr, yr, colw, "Analysis",
                  f"Baseline returns whole sentences, so EM is low ({pct(base['EM'])}) even when the right sentence is found "
                  f"({pct(base['Contains'])} contain the answer). The transformer reader returns a short span "
                  f"(EM {pct(main['EM'])}, F1 {pct(main['F1'])}). Remaining errors are listed in the report.", size)

    # ---- bottom, full width ----
    y = min(yl, yr)
    fw = W - 2 * M
    y = text_box(c, M, y, fw, "Real-World Applications & Limitations",
                 "<b>Uses:</b> study assistants, FAQ and helpdesk bots, document search, and the reader stage of RAG pipelines. "
                 "<b>Limits:</b> answer must exist as a span in the text; it cannot combine facts or explain; "
                 "a SQuAD-trained model can be confidently wrong; small test set.", size)
    qr_size = 44 if args.qr else 0
    cw = fw - (qr_size + GAP if args.qr else 0)
    y_after = text_box(c, M, y, cw, "Conclusion / Future Scope",
                       f"Result: EM <b>{pct(main['EM'])}</b>, F1 <b>{pct(main['F1'])}</b> vs baseline EM {pct(base['EM'])}, F1 {pct(base['F1'])}. "
                       "<b>Next:</b> dense retrieval (sentence embeddings), SQuAD 2.0 no-answer handling, fine-tuning on course Q&amp;A, "
                       "generative RAG with citations.", size)
    if args.qr:
        d = Drawing(qr_size, qr_size)
        widget = qr.QrCodeWidget(args.qr)
        x0, y0, x1, y1 = widget.getBounds()
        d.add(widget)
        d.transform = (qr_size / (x1 - x0), 0, 0, qr_size / (y1 - y0), 0, 0)
        renderPDF.draw(d, c, W - M - qr_size, y_after + GAP)
        c.setFont("Helvetica", size - 1)
        c.setFillColor(INK)
        c.drawCentredString(W - M - qr_size / 2, y_after + GAP - 6, "Scan: code / demo")
    return y_after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--out", default="poster.pdf")
    ap.add_argument("--title", default="Extractive Question Answering on Course Material using a Pretrained Transformer Encoder")
    ap.add_argument("--team", default="Member 1, Member 2, Member 3, Member 4")
    ap.add_argument("--qr", default=None, help="optional URL (GitHub repo / demo video) to encode as a QR code")
    args = ap.parse_args()

    R = json.load(open(args.results, encoding="utf-8"))
    if R["main_system"].startswith("baseline"):
        sys.exit("results.json only contains the baseline. Run  python evaluate.py  (without --no-model) first.")
    if args.team.startswith("Member 1"):
        print("Reminder: pass your names with --team \"A, B, C, D\"")

    # pick the largest font size at which everything fits on the page
    for size in (7.4, 7.2, 7.0, 6.8, 6.6, 6.4, 6.2, 6.0, 5.8):
        probe = canvas.Canvas(io.BytesIO(), pagesize=A5)
        if draw(probe, size, args, R) >= M:
            break
    c = canvas.Canvas(args.out, pagesize=A5)
    c.setTitle(args.title)
    draw(c, size, args, R)
    c.save()
    print(f"Saved {args.out}  (body font {size} pt, A5)")


if __name__ == "__main__":
    main()
