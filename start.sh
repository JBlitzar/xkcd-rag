#!/bin/bash
nohup ollama serve > /dev/null 2>&1 &
cd ~/xkcd-rag
~/.local/bin/uv run python bootstrap_cache.py || echo "Cache bootstrap failed; continuing with whatever is on disk."
while true; do
    ~/.local/bin/uv run bot.py
    echo "Bot crashed, restarting in 10 seconds..."
    sleep 10
done