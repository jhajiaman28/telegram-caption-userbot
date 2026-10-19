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

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Telegram Caption Userbot is Active and Running 24/7!")

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()

    def log_message(self, format, *args):
        pass

def start_health_server():
    port = int(os.getenv("PORT", "8080"))
    try:
        server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"Healthcheck HTTP server listening on port {port}.")
    except Exception as e:
        logger.warning(f"Could not bind HTTP healthcheck on port {port}: {e}")

def init_userbot():
    config = UserbotConfig.from_env()

    if not config.api_id or not config.api_hash or not config.gemini_api_key:
        logger.error("Missing required environment credentials.")
        sys.exit(1)

    os.makedirs(config.download_dir, exist_ok=True)
    start_health_server()

    session_obj = StringSession(config.session_string) if config.session_string else config.session_name
    client = TelegramClient(session_obj, config.api_id, config.api_hash)
    captioner = GeminiCaptioner(config.gemini_api_key)

    async def process_topic_video(chat, topic_id: int, topic_title: str):
        logger.info(f"Scanning topic ID {topic_id} ('{topic_title}') for video...")

        status_msg = await client.send_message(
            chat,
            "⏳ <b>[Final] Detected:</b> Locating video in this topic to generate captions...",
            parse_mode="html",
            reply_to=topic_id
        )

        target_video_msg = None
        async for msg in client.iter_messages(chat, reply_to=topic_id, limit=40):
            if msg.video:
                target_video_msg = msg
                break
            if msg.media and hasattr(msg.media, "document"):
                doc = msg.media.document
                if hasattr(doc, "mime_type") and doc.mime_type.startswith("video/"):
                    target_video_msg = msg
                    break

        if not target_video_msg:
            await status_msg.edit(
                "⚠️ <b>Notice:</b> [Final] was detected, but no video file was found in this topic thread.",
                parse_mode="html"
            )
            return

        file_size_mb = (target_video_msg.file.size / (1024 * 1024)) if (target_video_msg.file and target_video_msg.file.size) else 0
        await status_msg.edit(
            f"📥 <b>Downloading Video ({file_size_mb:.1f} MB)...</b>\n<i>High-speed MTProto cloud stream.</i>",
            parse_mode="html"
        )

        video_path = await client.download_media(target_video_msg, file=config.download_dir)
        if not video_path or not os.path.exists(video_path):
            await status_msg.edit("❌ <b>Download Error:</b> Failed to download video file.", parse_mode="html")
            return

        try:
            await status_msg.edit(
                "🧠 <b>Analyzing Video with Gemini AI...</b>\n<i>Drafting 4 separate platform messages...</i>",
                parse_mode="html"
            )

            caption_blocks = captioner.generate_captions(video_path, topic_context=topic_title)
            if isinstance(caption_blocks, str):
                caption_blocks = [b.strip() for b in caption_blocks.split("===SPLIT_MSG===") if b.strip()]

            for block in caption_blocks:
                await client.send_message(
                    chat,
                    block,
                    reply_to=target_video_msg.id
                )
                await asyncio.sleep(0.5)

            await status_msg.delete()
            logger.info(f"Successfully posted {len(caption_blocks)} separate messages for topic {topic_id}!")

        except Exception as e:
            logger.error(f"Error generating captions: {e}", exc_info=True)
            await status_msg.edit(f"❌ <b>Caption Generation Failed:</b>\n<code>{e}</code>", parse_mode="html")

        finally:
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except Exception:
                    pass

    @client.on(events.Raw)
    async def raw_handler(event):
        if hasattr(event, "message") and isinstance(event.message, types.MessageService):
            action = getattr(event.message, "action", None)
            if isinstance(action, types.MessageActionTopicEdit):
                new_title = action.title or ""
                if config.trigger_keyword in new_title.lower():
                    topic_id = event.message.reply_to.reply_to_top_id or event.message.reply_to.reply_to_msg_id or event.message.id
                    chat = await event.message.get_chat()
                    asyncio.create_task(process_topic_video(chat, topic_id, new_title))

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
        asyncio.create_task(process_topic_video(chat, topic_id, "Manual Trigger"))

    return client

if __name__ == "__main__":
    userbot = init_userbot()
    userbot.start()
    print("\n🟢 Userbot is ONLINE and listening 24/7 for [Final] topics...")
    userbot.run_until_disconnected()
