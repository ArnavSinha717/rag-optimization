#!/bin/bash
cd /home/sinha/code/rag
# wait for test 1 to finish
while ! grep -q "MULTI-QUERY" eval/multiquery_test.log 2>/dev/null; do sleep 20; done
echo "=== test 1 done, starting test 2 ===" 
.venv/bin/python eval/metadata_filter_test.py > eval/metadata_filter_test.log 2>&1
echo "=== test 2 done, ingesting v08 (ctx recursive) ==="
.venv/bin/python eval/contextual_ingest.py v04 > eval/ctx_ingest_v08.log 2>&1
echo "=== ingesting v09 (ctx markdown) ==="
.venv/bin/python eval/contextual_ingest.py v07 > eval/ctx_ingest_v09.log 2>&1
sleep 60   # let Atlas index the new chunks
echo "=== evaluating v08 ==="
.venv/bin/python eval/contextual_eval.py v08_ctx_recursive > eval/ctx_eval_v08.log 2>&1
echo "=== evaluating v09 ==="
.venv/bin/python eval/contextual_eval.py v09_ctx_markdown > eval/ctx_eval_v09.log 2>&1
echo "=== QUEUE COMPLETE ==="
