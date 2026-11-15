#!/bin/sh
# Start llama-server for the model named in $MODEL (original | broken | targeted20).
# --override-kv makes <EOS_TOKEN> (id 3) the end-of-text token: the original Tiny Aya ends answers with it, and without this
#   it runs on into junk (the stop-token finding). The add-on models stop at <|END_OF_TURN_TOKEN|> either way.
# --chat-template-file: the training tokenizer's template, the same one used for every judged answer.
# -n 400: default answer length cap; -c 2048: one conversation at a time (Lambda sends one request per instance).
set -e
case "$MODEL" in original|broken|targeted20) ;; *) echo "MODEL must be original, broken or targeted20" >&2; exit 1;; esac
exec /opt/llama-server -m /opt/models/$MODEL.gguf --host 0.0.0.0 --port 8080 -c 2048 -np 1 -t 6 -n 400 \
  --jinja --chat-template-file /opt/chat_template.jinja --override-kv tokenizer.ggml.eos_token_id=int:3 --no-webui
