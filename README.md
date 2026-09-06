# xkcd-rag

XKCD RAG Discord bot, containerized.

```bash
make up      # docker compose up -d --build
make logs    # docker compose logs -f
make down
```

`DISCORD_TOKEN` and tuning vars go in `.env` (gitignored).

## Caches

`embeddings_cache/`, `explainxkcd_cache/`, and `onnx_cache/` are gitignored.
`bootstrap_cache.py` downloads them from the GitHub Release `cache-latest` on
first run (or `--force` to re-download). The entrypoint runs it automatically.

## Cache refresh (cron)

A GitHub Action rebuilds `cache.tar.gz` daily at 19:00 UTC. A host-side
systemd timer on the Pi pulls the fresh cache and restarts the container:

```bash
# one-time install
cp cache-refresh.service cache-refresh.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now cache-refresh.timer

# manual refresh
systemctl --user start cache-refresh.service

# check schedule
systemctl --user list-timers cache-refresh.timer
```
