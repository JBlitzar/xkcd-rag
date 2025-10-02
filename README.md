# xkcd-rag

XKCD RAG.

`uv sync --extra build-cache`

```bash
cp discord_bot.service ~/.config/systemd/user/discord-bot.service
systemctl --user daemon-reload
systemctl --user enable discord-bot
systemctl --user restart discord-bot
systemctl --user status discord-bot
loginctl enable-linger $USER
```

Logs at `journalctl --user -u discord-bot -f`.

or `journalctl --user -u discord-bot -n 50`.
