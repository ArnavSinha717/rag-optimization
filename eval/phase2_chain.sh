#!/bin/bash
cd /home/sinha/code/rag
while pgrep -f ingest_corpus50.py > /dev/null; do sleep 60; done
echo "=== ingest done, launching sweep ==="
.venv/bin/python -u eval/phase2_sweep.py
