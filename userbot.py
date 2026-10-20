import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging
import os
import shutil
import sys
import threading
import time
from telethon import TelegramClient, events, functions, types
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

# ------------------------------------------------------------------------------
# Userbot Core Logic
# ------------------------------------------------------------------------------
def init_userbot():
    config = UserbotConfig.from_env()

    if not config.api_id or not config.api_hash:
        logger.error("TELEGRAM_API_ID or TELEGRAM_API_HASH is missing.")
        sys.exit(1)

    if not config.gemini_api_key:
        logger.error("GEMINI_API_KEY is missing.")
        sys.exit(1)

    os.makedirs(config.download_dir, exist_ok=True)
    start_health_server()

    if config.session_string:
        logger.info("Initializing Telethon in Cloud Stateless Mode (StringSession)...")
        session_obj = StringSession(config.session_string)
    else:
        logger.info(f"Initializing Telethon in Local File Mode ({config.session_name}.session)...")
        session_obj = config.session_name

    client = TelegramClient(session_obj, config.api_id, config.api_hash)
    captioner = GeminiCaptioner(config.gemini_api_key)

    active_processing_topics = set()

    def is_final_title(title: str) -> bool:
        if not title:
            return False
        t = title.lower()
        return "[final]" in t or "(final)" in t or t.endswith(" final") or " final " in t

    async def get_topic_title(chat, topic_id: int) -> str:
        if not topic_id:
            return ""
        try:
            msg = await client.get_messages(chat, ids=topic_id)
            if msg and hasattr(msg, "action") and hasattr(msg.action, "title"):
                return msg.action.title
        except Exception:
            pass

        try:
            res = await client(functions.channels.GetForumTopicsByIDRequest(
                channel=chat,
                topics=[topic_id]
            ))
            if res and hasattr(res, "topics") and res.topics:
                return res.topics[0].title
        except Exception:
            pass

        return ""

    def is_message_image(msg) -> bool:
        if not msg:
            return False
        if msg.photo:
            return True
        if msg.media and hasattr(msg.media, "document"):
            doc = msg.media.document
            if hasattr(doc, "mime_type") and doc.mime_type.startswith("image/"):
                return True
        return False

    async def process_topic_video(chat, topic_id: int, topic_title: str, target_video_msg=None):
        if topic_id in active_processing_topics:
            logger.info(f"Topic {topic_id} is already being processed. Skipping duplicate trigger.")
            return

        active_processing_topics.add(topic_id)
        logger.info(f"Processing video for topic ID {topic_id} ('{topic_title}')...")

        status_msg = None
        try:
            status_msg = await client.send_message(
                chat,
                "⏳ <b>[Final] Detected:</b> Locating video in this topic to generate captions...",
                parse_mode="html",
                reply_to=topic_id
            )

            if not target_video_msg:
                async for msg in client.iter_messages(chat, reply_to=topic_id, limit=50):
                    if msg.video:
                        target_video_msg = msg
                        break
                    if msg.media and hasattr(msg.media, "document"):
                        doc = msg.media.document
                        if hasattr(doc, "mime_type") and doc.mime_type.startswith("video/"):
                            target_video_msg = msg
                            break

            if not target_video_msg:
                logger.warning(f"No video found in topic thread {topic_id}.")
                if status_msg:
                    await status_msg.edit(
                        "⚠️ <b>Notice:</b> [Final] was detected, but no video file was found in this topic thread.",
                        parse_mode="html"
                    )
                return

            file_size_mb = 0
            if target_video_msg.file and target_video_msg.file.size:
                file_size_mb = target_video_msg.file.size / (1024 * 1024)

            logger.info(f"Video found (Message ID {target_video_msg.id}, Size: {file_size_mb:.2f} MB). Downloading...")
            if status_msg:
                await status_msg.edit(
                    f"📥 <b>Downloading Video ({file_size_mb:.1f} MB)...</b>\n<i>High-speed MTProto cloud stream.</i>",
                    parse_mode="html"
                )

            video_path = await client.download_media(target_video_msg, file=config.download_dir)
            if not video_path or not os.path.exists(video_path):
                if status_msg:
                    await status_msg.edit("❌ <b>Download Error:</b> Failed to download video file.", parse_mode="html")
                return

            try:
                if status_msg:
                    await status_msg.edit(
                        "🧠 <b>Analyzing Video with Gemini AI...</b>\n<i>Drafting 4 separate platform messages...</i>",
                        parse_mode="html"
                    )

                raw_captions = captioner.generate_captions(video_path, topic_context=topic_title)
                
                if isinstance(raw_captions, str):
                    caption_blocks = [b.strip() for b in raw_captions.split("===SPLIT_MSG===") if b.strip()]
                else:
                    caption_blocks = raw_captions

                for block in caption_blocks:
                    await client.send_message(
                        chat,
                        block,
                        reply_to=target_video_msg.id
                    )
                    await asyncio.sleep(0.5)

                if status_msg:
                    await status_msg.delete()
                logger.info(f"Successfully posted {len(caption_blocks)} separate caption messages for topic {topic_id}!")

            except Exception as gen_err:
                logger.error(f"Error generating captions: {gen_err}", exc_info=True)
                if status_msg:
                    await status_msg.edit(f"❌ <b>Caption Generation Failed:</b>\n<code>{gen_err}</code>", parse_mode="html")

            finally:
                if os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                        logger.info(f"Cleaned up temp video file: {video_path}")
                    except Exception:
                        pass

        except Exception as e:
            logger.error(f"Unexpected error in process_topic_video: {e}", exc_info=True)
        finally:
            active_processing_topics.discard(topic_id)

    async def process_topic_photos(chat, topic_id: int, topic_title: str, photo_messages: list):
        if not photo_messages:
            return

        status_msg = None
        temp_folder = os.path.join(config.download_dir, f"album_{topic_id}_{int(time.time())}")
        os.makedirs(temp_folder, exist_ok=True)
        downloaded_paths = []

        try:
            status_msg = await client.send_message(
                chat,
                f"⏳ <b>Found {len(photo_messages)} Photos!</b> Downloading & analyzing with Gemini AI...",
                parse_mode="html",
                reply_to=topic_id
            )

            for p_msg in photo_messages:
                p_path = await client.download_media(p_msg, file=temp_folder)
                if p_path and os.path.exists(p_path):
                    downloaded_paths.append(p_path)

            if not downloaded_paths:
                if status_msg:
                    await status_msg.edit("❌ <b>Download Error:</b> Failed to download photos.", parse_mode="html")
                return

            if status_msg:
                await status_msg.edit(
                    f"🧠 <b>Analyzing {len(downloaded_paths)} Photos with Gemini AI...</b>\n<i>Drafting unified album captions...</i>",
                    parse_mode="html"
                )

            raw_captions = captioner.generate_multi_photo_captions(downloaded_paths, topic_context=topic_title)
            
            if isinstance(raw_captions, str):
                caption_blocks = [b.strip() for b in raw_captions.split("===SPLIT_MSG===") if b.strip()]
            else:
                caption_blocks = raw_captions

            reply_target_id = photo_messages[0].id
            for block in caption_blocks:
                await client.send_message(
                    chat,
                    block,
                    reply_to=reply_target_id
                )
                await asyncio.sleep(0.5)

            if status_msg:
                await status_msg.delete()
            logger.info(f"Successfully posted {len(caption_blocks)} photo album caption messages!")

        except Exception as e:
            logger.error(f"Error generating photo album captions: {e}", exc_info=True)
            if status_msg:
                await status_msg.edit(f"❌ <b>Photo Caption Generation Failed:</b>\n<code>{e}</code>", parse_mode="html")

        finally:
            if os.path.exists(temp_folder):
                try:
                    shutil.rmtree(temp_folder)
                    logger.info(f"Cleaned up temp photo album folder: {temp_folder}")
                except Exception:
                    pass

    # --------------------------------------------------------------------------
    # Event 1: NewMessage Listener (Video Upload + Topic Rename)
    # --------------------------------------------------------------------------
    @client.on(events.NewMessage)
    async def message_handler(event):
        is_video = bool(
            event.message.video or (
                event.message.media and hasattr(event.message.media, "document") and
                getattr(event.message.media.document, "mime_type", "").startswith("video/")
            )
        )

        if is_video and event.message.reply_to:
            topic_id = getattr(event.message.reply_to, "reply_to_top_id", None) or getattr(event.message.reply_to, "reply_to_msg_id", None)
            if topic_id:
                chat = await event.get_chat()
                topic_title = await get_topic_title(chat, topic_id)
                logger.info(f"New video uploaded in topic {topic_id} ('{topic_title}')")
                if is_final_title(topic_title):
                    logger.info(f"Video uploaded to [Final] topic {topic_id}! Triggering captions immediately...")
                    asyncio.create_task(process_topic_video(chat, topic_id, topic_title, target_video_msg=event.message))
                    return

        action = getattr(event.message, "action", None)
        if isinstance(action, types.MessageActionTopicEdit):
            new_title = action.title or ""
            logger.info(f"Topic rename detected via NewMessage: '{new_title}'")
            if is_final_title(new_title):
                topic_id = None
                if event.message.reply_to:
                    topic_id = getattr(event.message.reply_to, "reply_to_top_id", None) or getattr(event.message.reply_to, "reply_to_msg_id", None)
                if not topic_id:
                    topic_id = event.message.id

                chat = await event.get_chat()
                logger.info(f"Triggering auto-captions on rename for topic {topic_id} in {chat.id} ('{new_title}')")
                asyncio.create_task(process_topic_video(chat, topic_id, new_title))

    # --------------------------------------------------------------------------
    # Event 2: Manual Command (.final / /final) for Single Video
    # --------------------------------------------------------------------------
    @client.on(events.NewMessage(pattern=r"(?i)^[./]final"))
    async def manual_trigger(event):
        chat = await event.get_chat()
        topic_id = None
        if event.message.reply_to:
            topic_id = getattr(event.message.reply_to, "reply_to_top_id", None) or getattr(event.message.reply_to, "reply_to_msg_id", None)

        if not topic_id:
            topic_id = getattr(event.message, "reply_to_msg_id", None) or event.message.id

        logger.info(f"Manual .final trigger received in topic {topic_id}")
        asyncio.create_task(process_topic_video(chat, topic_id, "Manual Trigger"))

    # --------------------------------------------------------------------------
    # Event 3: Manual Command (.photos / .photo / .album / .images) for Multiple Photos
    # --------------------------------------------------------------------------
    @client.on(events.NewMessage(pattern=r"(?i)^[./](photos?|album|images?)(?:\s|$)"))
    async def manual_photos_trigger(event):
        chat = await event.get_chat()
        topic_id = None
        if event.message.reply_to:
            topic_id = getattr(event.message.reply_to, "reply_to_top_id", None) or getattr(event.message.reply_to, "reply_to_msg_id", None)

        if not topic_id:
            topic_id = getattr(event.message, "reply_to_msg_id", None) or event.message.id

        logger.info(f"Manual .photos command triggered in topic {topic_id}")
        
        # Immediate status acknowledgment so Telegram is never silent
        ack_msg = await client.send_message(
            chat,
            "🔍 <b>Searching for photos in this topic...</b>",
            parse_mode="html",
            reply_to=topic_id
        )

        try:
            topic_title = await get_topic_title(chat, topic_id) or "Photo Album"
            reply_msg = await event.get_reply_message()
            photo_messages = []

            # 1. If replying directly to an album with a grouped_id, collect all photos with that grouped_id
            if reply_msg and getattr(reply_msg, "grouped_id", None):
                target_gid = reply_msg.grouped_id
                async for m in client.iter_messages(chat, reply_to=topic_id, limit=40):
                    if getattr(m, "grouped_id", None) == target_gid and is_message_image(m):
                        photo_messages.append(m)
            # 2. If replying directly to a single photo
            elif reply_msg and is_message_image(reply_msg):
                photo_messages.append(reply_msg)
                async for m in client.iter_messages(chat, reply_to=topic_id, limit=20):
                    if is_message_image(m) and m.id != reply_msg.id:
                        photo_messages.append(m)
                    if len(photo_messages) >= 10:
                        break
            # 3. If sent standalone in the topic thread, collect recent photos
            else:
                async for m in client.iter_messages(chat, reply_to=topic_id, limit=40):
                    if is_message_image(m):
                        photo_messages.append(m)
                    if len(photo_messages) >= 10:
                        break

            if not photo_messages:
                await ack_msg.edit(
                    "⚠️ <b>No photos found:</b> Please use <code>.photos</code> in a topic that contains uploaded photos, or reply directly to a photo or album.",
                    parse_mode="html"
                )
                return

            await ack_msg.delete()
            photo_messages.reverse()  # Chronological order
            asyncio.create_task(process_topic_photos(chat, topic_id, topic_title, photo_messages))

        except Exception as e:
            logger.error(f"Error in manual_photos_trigger: {e}", exc_info=True)
            await ack_msg.edit(f"❌ <b>Error:</b> <code>{e}</code>", parse_mode="html")

    return client

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Telegram Userbot (Render Free Web Service Ready)")
    print("="*60)
    userbot = init_userbot()
    userbot.start()
    print("\n🟢 Userbot is ONLINE and listening 24/7 for [Final] topics & .photos...")
    userbot.run_until_disconnected()
