# xkcd-rag
XKCD RAG.



```bash
cp discord_bot.service ~/.config/systemd/user/discord-bot.service
systemctl --user daemon-reload
systemctl --user enable discord-bot
systemctl --user start discord-bot
systemctl --user status discord-bot
loginctl enable-linger $USER
```

Logs at `journalctl --user -u discord-bot -f`.