import os
import asyncio
import logging
import json
import time
from typing import List, Dict, Optional
import dotenv
import discord

from embedding_search import query_xkcd, getQuantizedEmbedder

dotenv.load_dotenv()
getQuantizedEmbedder()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("xkcd-bot")


DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
MSG_HISTORY_COUNT = int(os.environ.get("XKCD_MSG_COUNT", "10"))
SCORE_THRESHOLD = float(os.environ.get("XKCD_SCORE_THRESHOLD", "0.64"))
ACTIVATION_COUNT = int(os.environ.get("XKCD_ACTIVATION_COUNT", "30"))
COUNTERS_FILE = "channel_counters.json"
HISTORY_FILE = "channel_xkcd_history.json"
REPEAT_WINDOW_DAYS = float(os.environ.get("XKCD_REPEAT_WINDOW_DAYS", "14"))
REPEAT_WINDOW_SECONDS = int(REPEAT_WINDOW_DAYS * 86400)

if not DISCORD_TOKEN:
    logger.warning(
        "DISCORD_TOKEN not set - bot will fail to connect until you set it in the environment"
    )


intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

client = discord.Client(intents=intents)

# queue of discord.Message objects to process
message_queue: asyncio.Queue[discord.Message] = asyncio.Queue()

# per-channel counters: number of user messages since last activation
channel_counters: Dict[int, int] = {}

# per-channel history of last sent time per comic_number (epoch seconds)
channel_history: Dict[int, Dict[int, float]] = {}


def _load_channel_counters_sync() -> Dict[int, int]:
    """Synchronous helper function to load channel counters from disk."""
    if os.path.exists(COUNTERS_FILE):
        with open(COUNTERS_FILE, "r") as f:
            data = json.load(f)
            # Convert string keys back to int (JSON keys are always strings)
            return {int(k): v for k, v in data.items()}
    return {}


async def load_channel_counters() -> Dict[int, int]:
    """Load channel counters from disk asynchronously. Returns empty dict if file doesn't exist or is invalid."""
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _load_channel_counters_sync)
    except (json.JSONDecodeError, ValueError, IOError) as e:
        logger.warning(f"Failed to load channel counters from {COUNTERS_FILE}: {e}")
        return {}


def _save_channel_counters_sync(counters: Dict[int, int]) -> None:
    """Synchronous helper function to save channel counters to disk."""
    with open(COUNTERS_FILE, "w") as f:
        json.dump(counters, f, indent=2)


async def save_channel_counters(counters: Dict[int, int]) -> None:
    """Save channel counters to disk asynchronously."""
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _save_channel_counters_sync, counters)
    except IOError as e:
        logger.error(f"Failed to save channel counters to {COUNTERS_FILE}: {e}")


def _prune_history_map(
    history: Dict[int, Dict[int, float]], now: Optional[float] = None
) -> Dict[int, Dict[int, float]]:
    """Return a new history with entries older than the repeat window removed."""
    now = now or time.time()
    cutoff = now - REPEAT_WINDOW_SECONDS
    pruned: Dict[int, Dict[int, float]] = {}
    for chan_id, mapping in history.items():
        if not isinstance(mapping, dict):
            continue
        filtered = {
            cn: ts
            for cn, ts in mapping.items()
            if isinstance(ts, (int, float)) and ts >= cutoff
        }
        if filtered:
            pruned[chan_id] = filtered
    return pruned


def _load_channel_history_sync() -> Dict[int, Dict[int, float]]:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            raw = json.load(f)
            out: Dict[int, Dict[int, float]] = {}
            for chan_str, mapping in raw.items():
                try:
                    chan_id = int(chan_str)
                except ValueError:
                    continue
                inner: Dict[int, float] = {}
                if isinstance(mapping, dict):
                    for comic_str, ts in mapping.items():
                        try:
                            inner[int(comic_str)] = float(ts)
                        except (ValueError, TypeError):
                            continue
                out[chan_id] = inner
            # Prune any entries older than the repeat window
            return _prune_history_map(out, time.time())
    return {}


async def load_channel_history() -> Dict[int, Dict[int, float]]:
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _load_channel_history_sync)
    except (json.JSONDecodeError, ValueError, IOError) as e:
        logger.warning(f"Failed to load channel history from {HISTORY_FILE}: {e}")
        return {}


def _save_channel_history_sync(history: Dict[int, Dict[int, float]]) -> None:
    # Prune old entries before persisting to disk
    pruned = _prune_history_map(history, time.time())
    serializable = {
        str(chan): {str(cn): ts for cn, ts in mapping.items()}
        for chan, mapping in pruned.items()
    }
    with open(HISTORY_FILE, "w") as f:
        json.dump(serializable, f, indent=2)


async def save_channel_history(history: Dict[int, Dict[int, float]]) -> None:
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _save_channel_history_sync, history)
    except IOError as e:
        logger.error(f"Failed to save channel history to {HISTORY_FILE}: {e}")


def is_recently_sent(
    chan_id: int, comic_number: int, now: Optional[float] = None
) -> bool:
    now = now or time.time()
    last = channel_history.get(chan_id, {}).get(comic_number)
    if last is None:
        return False
    return (now - last) < REPEAT_WINDOW_SECONDS


async def record_sent(chan_id: int, comic_number: int) -> None:
    ch = channel_history.get(chan_id)
    if ch is None:
        ch = {}
        channel_history[chan_id] = ch
    ch[comic_number] = time.time()
    await save_channel_history(channel_history)


async def fetch_last_user_messages(
    channel: discord.TextChannel, count: int, bot_user: discord.ClientUser
) -> List[str]:
    """Collect the last `count` messages in `channel` not sent by the bot itself.

    Returns messages in chronological order (oldest first).
    """
    msgs: List[str] = []
    # Walk history newest->oldest until we have `count` user messages or hit a reasonable limit
    async for m in channel.history(limit=500):
        if m.author.bot:
            continue
        if m.author == bot_user:
            continue
        if not m.content:
            continue
        msgs.append(m.content)
        if len(msgs) >= count:
            break
    msgs.reverse()
    return msgs


async def worker_loop():
    """Background worker that processes messages from the queue one at a time."""
    loop = asyncio.get_running_loop()
    while True:
        msg: discord.Message = await message_queue.get()
        try:
            channel = msg.channel
            chan_id = channel.id

            # Check if we should still process this message based on current counter state
            current_count = channel_counters.get(chan_id, 0)
            if current_count < ACTIVATION_COUNT:
                logger.debug(
                    f"Skipping queued message for channel {chan_id}: "
                    f"current count {current_count} < activation threshold {ACTIVATION_COUNT}"
                )
                continue

            # Determine query source: explicit `!xkcd {query}` or recent history
            explicit_query: Optional[str] = None
            if isinstance(msg.content, str) and msg.content.strip().lower().startswith(
                "!xkcd"
            ):
                # Everything after the command is treated as the query
                explicit_query = msg.content.strip()[5:].strip()

            if explicit_query:
                query_text = explicit_query
            else:
                # fetch last user messages including the triggering message
                messages = await fetch_last_user_messages(
                    channel, MSG_HISTORY_COUNT, client.user
                )
                if not messages:
                    logger.debug("No messages found to query")
                    continue
                query_text = "\n".join(messages)

            # run the potentially blocking query in threadpool with server_mode=True
            # fetch more candidates to allow skipping recently sent comics per channel
            results = await loop.run_in_executor(
                None, lambda: query_xkcd(query_text, 25, server_mode=True)
            )
            if not results:
                continue

            # pick first candidate not sent within the repeat window for this channel
            selected: Optional[tuple[int, str, float]] = None
            now_ts = time.time()
            for cn, expl, sc in results:
                if not is_recently_sent(chan_id, cn, now_ts):
                    selected = (cn, expl, sc)
                    break

            if selected is None:
                # All top candidates recently sent; don't repeat
                if explicit_query is not None and explicit_query != "":
                    try:
                        top_score = results[0][2]
                        await channel.send(
                            f"Top xkcd match score was {top_score:.2f}, but recent matches were already posted here."
                        )
                        channel_counters[chan_id] = 0
                        await save_channel_counters(channel_counters)
                    except Exception:
                        logger.exception("Failed to send repeat-notice message")
                else:
                    logger.info(
                        f"All top candidates are within repeat window for channel {chan_id}; not posting."
                    )
                continue

            comic_number, explanation, score = selected
            if score > SCORE_THRESHOLD:
                url = f"https://xkcd.com/{comic_number}/"
                try:
                    print("Sending!!")
                    await channel.send(f"Best xkcd match (score {score:.2f}): {url}")
                    # Reset counter only after successfully sending a message
                    channel_counters[chan_id] = 0
                    await save_channel_counters(channel_counters)
                    await record_sent(chan_id, comic_number)
                    logger.info(
                        f"Reset counter for channel {chan_id} after sending message"
                    )
                except Exception:
                    logger.exception("Failed to send message to channel")
            else:
                # For explicit queries, inform user of the highest score without linking
                if explicit_query is not None and explicit_query != "":
                    try:
                        await channel.send(
                            f"Top xkcd match score was {score:.2f} (below threshold {SCORE_THRESHOLD:.2f})."
                        )
                        channel_counters[chan_id] = 0
                        await save_channel_counters(channel_counters)
                        logger.info(
                            f"Reset counter for channel {chan_id} after sending below-threshold notice"
                        )
                    except Exception:
                        logger.exception("Failed to send below-threshold score message")
                else:
                    logger.debug(
                        f"Top match score {score:.3f} below threshold {SCORE_THRESHOLD}, not posting."
                    )

        except Exception:
            logger.exception("Error processing queued message")
        finally:
            message_queue.task_done()


@client.event
async def on_ready():
    global channel_counters, channel_history
    logger.info(f"Logged in as {client.user} (id: {client.user.id})")

    # Load persistent channel counters
    channel_counters = await load_channel_counters()
    logger.info(f"Loaded counters for {len(channel_counters)} channels")

    # Load per-channel comic history
    channel_history = await load_channel_history()
    logger.info(f"Loaded comic history for {len(channel_history)} channels")

    # start worker
    client.loop.create_task(worker_loop())


@client.event
async def on_message(message: discord.Message):
    # ignore messages from bots (including ourselves)
    if message.author.bot:
        return
    if not isinstance(message.channel, discord.TextChannel):
        # only monitor text channels
        return

    chan_id = message.channel.id

    # Check for !xkcd command (optionally followed by a query) - trigger immediately and reset cooldown
    content_stripped = message.content.strip()
    if content_stripped.lower().startswith("!xkcd"):
        try:
            message_queue.put_nowait(message)
            # Reset the counter immediately for !xkcd command
            channel_counters[chan_id] = ACTIVATION_COUNT
            await save_channel_counters(channel_counters)
            logger.info(
                f"!xkcd command received in channel {chan_id}; queued for immediate processing"
            )
        except asyncio.QueueFull:
            logger.warning("Message queue full; dropping !xkcd command message")
        return

    # Track per-channel user message counts and only enqueue when we've seen
    # ACTIVATION_COUNT user messages in that channel since the last time the
    # bot posted (the counter is reset only when a message is actually sent).
    # increment counter for this channel
    channel_counters[chan_id] = channel_counters.get(chan_id, 0) + 1

    # Save counters to disk after each update
    await save_channel_counters(channel_counters)

    # If we've hit the activation count, enqueue this message for processing
    # Counter will be reset only if a message is actually sent
    if channel_counters[chan_id] >= ACTIVATION_COUNT:
        try:
            message_queue.put_nowait(message)
            logger.info(
                f"Channel {chan_id} reached activation count ({ACTIVATION_COUNT}); queued message for processing"
            )
        except asyncio.QueueFull:
            # should not happen unless a maxsize is set; log and drop
            logger.warning("Message queue full; dropping activation message")


if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
    # Run bot
