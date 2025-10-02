#!/bin/bash

cp discord_bot.service ~/.config/systemd/user/discord-bot.service
systemctl --user daemon-reload
systemctl --user enable discord-bot
systemctl --user restart discord-bot
systemctl --user status discord-bot
loginctl enable-linger $USER

echo "if no work, try:"
echo "export XDG_RUNTIME_DIR=/run/user/3277"
echo "export DBUS_SESSION_BUS_ADDRESS=unix:path=\$XDG_RUNTIME_DIR/bus"
echo "To view logs:"
echo "journalctl --user -u discord-bot -f -n 50"