SERVICE := discord-bot

.PHONY: start stop restart status logs

start:
	systemctl --user start $(SERVICE)

stop:
	systemctl --user stop $(SERVICE)

restart:
	systemctl --user restart $(SERVICE)

status:
	systemctl --user status $(SERVICE) --no-pager -n 30

logs:
	journalctl --user -u $(SERVICE) -f -n 50
