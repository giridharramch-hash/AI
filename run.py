"""
One-Command Runner for AI Mini-Capstone Project (25CS301)
Extractive Question Answering on Course Material using a Pretrained Transformer Encoder

Usage:
    python run.py
"""
import os
import sys
import time
import webbrowser
import uvicorn

def main():
    port = 8000
    print("=" * 70)
    print("  Extractive QA on Course Material (AI Mini-Capstone 25CS301)")
    print(f"  Web application starting at: http://localhost:{port}")
    print("=" * 70)
    uvicorn.run("backend.app:app", host="127.0.0.1", port=port, reload=False)

if __name__ == "__main__":
    main()
