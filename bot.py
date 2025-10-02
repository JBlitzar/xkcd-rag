import os
import asyncio
import logging
import json
from typing import List, Dict
import dotenv
import discord

from embedding_search import query_xkcd, getQuantizedEmbedder

dotenv.load_dotenv()
getQuantizedEmbedder()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("xkcd-bot")


DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
MSG_HISTORY_COUNT = int(os.environ.get("XKCD_MSG_COUNT", "10"))
SCORE_THRESHOLD = float(os.environ.get("XKCD_SCORE_THRESHOLD", "0.6"))
ACTIVATION_COUNT = int(os.environ.get("XKCD_ACTIVATION_COUNT", "30"))
COUNTERS_FILE = "channel_counters.json"

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


def load_channel_counters() -> Dict[int, int]:
    """Load channel counters from disk. Returns empty dict if file doesn't exist or is invalid."""
    try:
        if os.path.exists(COUNTERS_FILE):
            with open(COUNTERS_FILE, "r") as f:
                data = json.load(f)
                # Convert string keys back to int (JSON keys are always strings)
                return {int(k): v for k, v in data.items()}
    except (json.JSONDecodeError, ValueError, IOError) as e:
        logger.warning(f"Failed to load channel counters from {COUNTERS_FILE}: {e}")
    return {}


def save_channel_counters(counters: Dict[int, int]) -> None:
    """Save channel counters to disk."""
    try:
        with open(COUNTERS_FILE, "w") as f:
            json.dump(counters, f, indent=2)
    except IOError as e:
        logger.error(f"Failed to save channel counters to {COUNTERS_FILE}: {e}")


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

            # fetch last user messages including the triggering message
            messages = await fetch_last_user_messages(
                channel, MSG_HISTORY_COUNT, client.user
            )
            if not messages:
                logger.debug("No messages found to query")
                continue

            query_text = "\n".join(messages)

            # run the potentially blocking query in threadpool with server_mode=True
            results = await loop.run_in_executor(
                None, lambda: query_xkcd(query_text, 1, server_mode=True)
            )
            if not results:
                continue

            comic_number, explanation, score = results[0]
            if score > SCORE_THRESHOLD:
                url = f"https://xkcd.com/{comic_number}/"
                try:
                    print("Sending!!")
                    await channel.send(f"Best xkcd match (score {score:.2f}): {url}")
                    # Reset counter only after successfully sending a message
                    channel_counters[chan_id] = 0
                    save_channel_counters(channel_counters)
                    logger.info(
                        f"Reset counter for channel {chan_id} after sending message"
                    )
                except Exception:
                    logger.exception("Failed to send message to channel")
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
    global channel_counters
    logger.info(f"Logged in as {client.user} (id: {client.user.id})")

    # Load persistent channel counters
    channel_counters = load_channel_counters()
    logger.info(f"Loaded counters for {len(channel_counters)} channels")

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

    # Track per-channel user message counts and only enqueue when we've seen
    # ACTIVATION_COUNT user messages in that channel since the last time the
    # bot posted (the counter is reset only when a message is actually sent).
    chan_id = message.channel.id
    # increment counter for this channel
    channel_counters[chan_id] = channel_counters.get(chan_id, 0) + 1

    # Save counters to disk after each update
    save_channel_counters(channel_counters)

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
