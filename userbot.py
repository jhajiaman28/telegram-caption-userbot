import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging
import os
import sys
import threading
from telethon import TelegramClient, events, types
from telethon.sessions import StringSession
from config import UserbotConfig
from gemini_captioner import GeminiCaptioner

logging.basicConfig(
    format="%(asctime)s - [%(levelname)s] - %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("userbot")

# ------------------------------------------------------------------------------
# Built-in Lightweight HTTP Healthcheck for Free Cloud Web Services (Render / Koyeb)
# ------------------------------------------------------------------------------
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Telegram Caption Userbot is Active and Running 24/7!")

    def log_message(self, format, *args):
        # Silence access logs to keep terminal output clean
        pass

def start_health_server():
    port = int(os.getenv("PORT", "8080"))
    try:
        server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"Healthcheck HTTP server listening on port {port} (Enables Free Web Service Mode).")
    except Exception as e:
        logger.warning(f"Could not bind HTTP healthcheck on port {port}: {e}")

# ------------------------------------------------------------------------------
# Userbot Core Logic
# ------------------------------------------------------------------------------
def init_userbot():
    config = UserbotConfig.from_env()

    if not config.api_id or not config.api_hash:
        logger.error("TELEGRAM_API_ID or TELEGRAM_API_HASH is missing.")
        print("\n❌ Error: Telegram API credentials not configured.")
        print("Please set TELEGRAM_API_ID and TELEGRAM_API_HASH in your environment variables.\n")
        sys.exit(1)

    if not config.gemini_api_key:
        logger.error("GEMINI_API_KEY is missing.")
        print("\n❌ Error: GEMINI_API_KEY is not configured.")
        print("Please set GEMINI_API_KEY in your environment variables.\n")
        sys.exit(1)

    os.makedirs(config.download_dir, exist_ok=True)

    # Start HTTP server so Render / Koyeb can deploy as a 100% Free Web Service
    start_health_server()

    # Cloud Stateless Mode (via StringSession) vs Local File Session Mode
    if config.session_string:
        logger.info("Initializing Telethon in Cloud Stateless Mode (StringSession)...")
        session_obj = StringSession(config.session_string)
    else:
        logger.info(f"Initializing Telethon in Local File Mode ({config.session_name}.session)...")
        session_obj = config.session_name

    client = TelegramClient(session_obj, config.api_id, config.api_hash)
    captioner = GeminiCaptioner(config.gemini_api_key)

    async def process_topic_video(chat, topic_id: int, topic_title: str):
        """Locates the latest video in the topic, downloads via MTProto, calls Gemini, and posts captions."""
        logger.info(f"Scanning topic ID {topic_id} ('{topic_title}') for video...")

        status_msg = await client.send_message(
            chat,
            "⏳ <b>[Final] Detected:</b> Locating video in this topic to generate captions...",
            parse_mode="html",
            reply_to=topic_id
        )

        target_video_msg = None

        # Fetch recent messages inside this forum topic thread
        async for msg in client.iter_messages(chat, reply_to=topic_id, limit=40):
            if msg.video:
                target_video_msg = msg
                break
            # Also check documents with video mime types
            if msg.media and hasattr(msg.media, "document"):
                doc = msg.media.document
                if hasattr(doc, "mime_type") and doc.mime_type.startswith("video/"):
                    target_video_msg = msg
                    break

        if not target_video_msg:
            logger.warning(f"No video found in topic thread {topic_id}.")
            await status_msg.edit(
                "⚠️ <b>Notice:</b> [Final] was detected, but no video file was found in this topic thread.",
                parse_mode="html"
            )
            return

        # Calculate file size
        file_size_mb = 0
        if target_video_msg.file and target_video_msg.file.size:
            file_size_mb = target_video_msg.file.size / (1024 * 1024)

        logger.info(f"Video found (Message ID {target_video_msg.id}, Size: {file_size_mb:.2f} MB). Downloading...")
        await status_msg.edit(
            f"📥 <b>Downloading Video ({file_size_mb:.1f} MB)...</b>\n<i>High-speed MTProto cloud stream.</i>",
            parse_mode="html"
        )

        # Download video to local disk (supports up to 2 GB)
        video_path = await client.download_media(target_video_msg, file=config.download_dir)
        if not video_path or not os.path.exists(video_path):
            await status_msg.edit("❌ <b>Download Error:</b> Failed to download video file.", parse_mode="html")
            return

        try:
            await status_msg.edit(
                "🧠 <b>Analyzing Video with Gemini AI...</b>\n<i>Extracting audio, core wisdom, and drafting captions...</i>",
                parse_mode="html"
            )

            # Generate the 4 captions
            captions_text = captioner.generate_captions(video_path, topic_context=topic_title)

            # Send the final 4-platform captions
            header = f"✨ <b>Captions Ready for Topic:</b> <i>{topic_title}</i>\n" + "─"*38 + "\n\n"
            full_response = header + captions_text

            await client.send_message(
                chat,
                full_response,
                parse_mode="md",
                reply_to=target_video_msg.id
            )

            await status_msg.delete()
            logger.info(f"Successfully posted captions for topic {topic_id}!")

        except Exception as e:
            logger.error(f"Error generating captions: {e}", exc_info=True)
            await status_msg.edit(f"❌ <b>Caption Generation Failed:</b>\n<code>{e}</code>", parse_mode="html")

        finally:
            # Clean up local video file immediately to prevent disk accumulation
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    logger.info(f"Cleaned up temp video file: {video_path}")
                except Exception as del_err:
                    logger.warning(f"Could not delete temp file: {del_err}")

    # Event 1: Automatically detect when a topic is renamed
    @client.on(events.Raw)
    async def raw_handler(event):
        # Listen for topic edit actions in supergroups
        if isinstance(event, types.UpdateChannel):
            pass
        elif hasattr(event, "message") and isinstance(event.message, types.MessageService):
            action = getattr(event.message, "action", None)
            if isinstance(action, types.MessageActionTopicEdit):
                new_title = action.title or ""
                logger.info(f"Topic edit event detected: '{new_title}'")
                if config.trigger_keyword in new_title.lower():
                    # Identify the forum topic ID
                    topic_id = event.message.reply_to.reply_to_top_id or event.message.reply_to.reply_to_msg_id or event.message.id
                    chat = await event.message.get_chat()
                    asyncio.create_task(process_topic_video(chat, topic_id, new_title))

    # Event 2: Manual trigger shortcut (typing .final or /final inside the topic)
    @client.on(events.NewMessage(pattern=r"(?i)^[./]final"))
    async def manual_trigger(event):
        if not event.is_group:
            return

        topic_id = None
        if event.message.reply_to:
            topic_id = event.message.reply_to.reply_to_top_id or event.message.reply_to.reply_to_msg_id

        if not topic_id:
            await event.reply("⚠️ Please use this command inside a specific forum topic thread.")
            return

        chat = await event.get_chat()
        logger.info(f"Manual .final trigger received in topic {topic_id}")
        asyncio.create_task(process_topic_video(chat, topic_id, "Manual Trigger"))

    return client

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Telegram Userbot (Render Free Web Service Ready)")
    print("="*60)
    userbot = init_userbot()
    userbot.start()
    print("\n🟢 Userbot is ONLINE and listening 24/7 for [Final] topics...")
    userbot.run_until_disconnected()
