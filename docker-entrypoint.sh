#!/bin/sh
set -e
python bootstrap_cache.py || echo "Cache bootstrap failed; continuing with whatever is on disk."
exec python bot.py
