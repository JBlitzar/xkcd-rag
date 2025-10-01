#!/bin/bash
nohup ollama serve > /dev/null 2>&1 &
cd /home/jblitzar/xkcd-rag
while true; do
    /home/jblitzar/.local/bin/uv run bot.py
    echo "Bot crashed, restarting in 10 seconds..."
    sleep 10
done