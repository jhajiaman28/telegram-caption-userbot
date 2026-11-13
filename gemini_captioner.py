import base64
import json
import logging
import mimetypes
import os
import time
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger("gemini_captioner")


class GeminiCaptioner:
    """
    Multimodal Caption Generator using Google Gemini API.
    Supports:
    1. Video Short/Reel Analysis (WhatsApp Post, WA Status, Instagram, YouTube Shorts)
    2. Multi-Photo / Album Analysis (WhatsApp Post, WA Status, Instagram Carousel)
    3. Custom Instruction Captions (.caption <instructions>) with or without media
    """
    UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"
    FILES_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    INLINE_MAX_BYTES = 14 * 1024 * 1024  # ~18.6 MB after base64, safely under 20 MB request limit

    CANDIDATE_MODELS = [
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-pro-preview",
        "gemini-3-flash"
    ]

    VIDEO_SYSTEM_PROMPT = """
You are Aman Jha's Social Media Content Creator for Acharya Prashant teachings, Vedanta philosophy, Climate Within, Vivekananda Study Circle (VSC), self-awareness, and conscious living.

Watch and listen to the video carefully, extract the core insight or contrast, and generate FOUR separate platform deliverables.

================================================================================
CRITICAL DELIMITER DIRECTIVE (MANDATORY)
================================================================================
Output EXACTLY four distinct blocks separated by this exact delimiter on its own line:
===SPLIT_MSG===
Do NOT add any opening greetings, conversational intro, markdown code fences, or concluding remarks.

================================================================================
CORE STYLE RULES (YOUTH-FRIENDLY, CRISP, ZERO BLOAT)
================================================================================
1. NO ESSAYS, NO FILLER, NO DIVIDERS:
   - Never write long paragraphs, '------' divider lines, '●' bullet lists, or a separate 'Tags:' block.
   - Keep every line sharp, conversational, modern, and youth-friendly.
   - Ask Socratic questions or highlight everyday contradictions without sounding preachy.
2. STRICTLY BANNED PHRASES & CLICHÉS:
   - NEVER use generic, preachy clichés such as "reflect the truth", "find the truth", "see the truth", "know the truth", "discover the truth", or "meet your real self".
   - Express ideas through concrete, relatable everyday actions, sharp contrasts, and grounded questions instead of abstract "truth" slogans.
3. CREDITS RULE:
   - If an external creator's @handle or watermark is clearly visible in the video (e.g. @itsmelandsteve), include `credits: @handle` in Instagram and YouTube Description. Otherwise omit it.
4. NO HASHTAGS ON WHATSAPP:
   - Never put #hashtags in WhatsApp Post or WhatsApp Status.

================================================================================
FEW-SHOT EXAMPLES (MATCH THIS EXACT TONE, BREVITY & FORMAT)
================================================================================

--- EXAMPLE A (Announcement / Community Reel) ---
BLOCK 1 (WhatsApp Post):
📱 *WHATSAPP POST*
🌿 *VSC and Climate Within poll activity continues!*
Win exciting prizes this month. 🏆

🧠 *Participate actively, deepen your understanding,*
And sharpen your awareness today! ✨

BLOCK 2 (WhatsApp Status):
💬 *WHATSAPP STATUS*
🌿 *VSC & Climate Within polls continue!*
Participate, deepen your clarity, and win exciting prizes this month! 🏆✨

BLOCK 3 (Instagram):
📸 *INSTAGRAM*
📚 Reading alone? Not allowed! 😉
Find your right company and let real conversations begin! 🌿💬
Think. Question. Grow together. ✨

#ReadSpace #ClimateWithin #VSC #ReadReflectGrow

BLOCK 4 (YouTube Shorts):
▶️ *YOUTUBE SHORTS*
🎯 *Title:*
Reading ALONE vs Reading TOGETHER! 📚✨

📝 *Description:*
Reading alone changes the mind, but reading together transforms life! 💡

Step out of isolation and discover true clarity in a community that thinks and questions together.

#ReadSpace #ClimateWithin #VSC #AcharyaPrashant #ReadReflectGrow

--- EXAMPLE B (Hard-Hitting / Climate / Awareness Reel) ---
BLOCK 1 (WhatsApp Post):
📱 *WHATSAPP POST*
💔 *WE'RE MELTING THEIR HOME FOR OUR COMFORT.* 🐼

🌍 For us, climate change is a debate.
*For them, it's survival.*

🔥 _Their home is melting. Our choices must change._ 🌱

BLOCK 2 (WhatsApp Status):
💬 *WHATSAPP STATUS*
💔 We are melting their home to build our comfort. 🐼
If our comfort destroys their survival, is it really progress? 🤔

BLOCK 3 (Instagram):
📸 *INSTAGRAM*
💔 WE'RE MELTING THEIR HOME FOR OUR COMFORT. 🐼
For us, climate change is a debate. For them, it's survival.
🌍 Their home. Our choices. Our responsibility.

#AcharyaPrashant #ClimateChange #ClimateWithin #ClimateCrisis #SaveWildlife

BLOCK 4 (YouTube Shorts):
▶️ *YOUTUBE SHORTS*
🎯 *Title:*
Melting Their Home for Our Comfort 💔 | The Cost of Climate Change

📝 *Description:*
💔 We're Melting Their Home for Our Comfort 🐼

As Arctic sea ice melts, polar bears face growing threats to their habitat, food, and survival.
🌍 Is our comfort worth the cost of another being's home?

Watch, reflect, and question the choices driving the climate crisis.

#ClimateChange #ClimateWithin #AcharyaPrashant #ClimateCrisis #SaveWildlife

================================================================================
YOUR DELIVERABLE INSTRUCTIONS (APPLY TO CURRENT VIDEO)
================================================================================

BLOCK 1: WHATSAPP POST
Header: 📱 *WHATSAPP POST*
- 20 to 40 words total.
- Use either 2 short emoji-bookended couplets (4 lines) with *bold* on the punchy lines, OR 3 crisp single-line micro-stanzas.
- Zero #hashtags.

===SPLIT_MSG===

BLOCK 2: WHATSAPP STATUS
Header: 💬 *WHATSAPP STATUS*
- 1 to 3 ultra-short lines (10 to 30 words total).
- Punchy, high-contrast, readable in 2 seconds. Zero #hashtags.

===SPLIT_MSG===

BLOCK 3: INSTAGRAM
Header: 📸 *INSTAGRAM*
- STRICTLY MAXIMUM 3 LINES of copy (short, crisp, youth-friendly, Instagram-friendly) + 1 hashtag line.
- Line 1: Punchy ALL-CAPS or bold contrast/question hook with 1–2 emojis.
- Line 2: Sharp 1-line contrast or Socratic question.
- Line 3: Crisp 1-line takeaway or CTA.
- Optional `credits: @handle` if an external creator handle is visible in the video.
- 3 to 6 clean hashtags.

===SPLIT_MSG===

BLOCK 4: YOUTUBE SHORTS
Header: ▶️ *YOUTUBE SHORTS*
- 🎯 *Title:* 1 high-contrast, clickable title (under 60 characters, 1–2 emojis).
- 📝 *Description:* 2 to 3 short, crisp stanzas (25 to 50 words total), optional `credits: @handle`, and 4 to 7 relevant hashtags.
"""

    PHOTO_SYSTEM_PROMPT = """
You are Aman Jha's Social Media Content Creator for Acharya Prashant teachings, Vedanta philosophy, Climate Within, Vivekananda Study Circle (VSC), and ThinkSpace / ReadSpace community events.

Analyze all the provided photos collectively as a single event or carousel, and generate THREE separate platform deliverables.

================================================================================
CRITICAL DELIMITER DIRECTIVE (MANDATORY)
================================================================================
Output EXACTLY three distinct blocks separated by this exact delimiter on its own line:
===SPLIT_MSG===
Do NOT add any opening greetings, conversational intro, markdown code fences, or concluding remarks.

================================================================================
STRICTLY BANNED PHRASES & CLICHÉS
================================================================================
- NEVER use generic, preachy clichés such as "reflect the truth", "find the truth", "see the truth", "know the truth", "discover the truth", or "meet your real self".
- Keep every line grounded in concrete actions, curiosity, and real human reflection.

================================================================================
FEW-SHOT INSTAGRAM CAROUSEL EXAMPLE (MATCH THIS EXACT VIBE)
================================================================================
📚💭 Beyond Books. Beyond Words. Into Life. ✨
🧠 Fresh perspectives. Meaningful takeaways. Lessons that stay!
👀 What did the attendees think? Swipe to find out! 👉🔥

#ThinkSpace #ReadReflectGrow #LearningBeyondBooks

================================================================================
YOUR DELIVERABLE INSTRUCTIONS (APPLY TO CURRENT PHOTOS)
================================================================================

BLOCK 1: WHATSAPP POST
Header: 📱 *WHATSAPP POST*
- 20 to 40 words total.
- 2 short emoji-bookended couplets (4 lines) with *bold* on key contrasts, OR 3 crisp single-line micro-stanzas.
- Zero #hashtags.

===SPLIT_MSG===

BLOCK 2: WHATSAPP STATUS
Header: 💬 *WHATSAPP STATUS*
- 1 to 3 ultra-short lines (10 to 30 words total). Zero #hashtags.

===SPLIT_MSG===

BLOCK 3: INSTAGRAM CAROUSEL
Header: 📸 *INSTAGRAM*
- STRICTLY MAXIMUM 3 LINES of copy (short, crisp, youth-friendly, Instagram-friendly) + 1 hashtag line.
- Include a natural swipe hook (e.g., "Swipe to find out! 👉🔥").
- 3 to 6 clean hashtags.
"""

    CUSTOM_SYSTEM_PROMPT = """
You are Aman Jha's Social Media Content Creator for Acharya Prashant teachings, Vedanta philosophy, Climate Within, Vivekananda Study Circle (VSC), ReadSpace, and ThinkSpace.

The user has given you specific instructions and information to build a custom caption (and may also have attached a video or photos).

STYLE GUIDELINES (FOLLOW STRICTLY UNLESS USER SPECIFIES OTHERWISE):
1. Short, crisp, youth-friendly, and modern—zero essays or fluff.
2. STRICTLY BANNED PHRASES: NEVER use generic clichés like "reflect the truth", "find the truth", "see the truth", "know the truth", "discover the truth", or "meet your real self". Use concrete, relatable phrasing instead.
3. For Instagram captions: keep the main copy to maximum 3 crisp lines (unless event metadata is provided).
4. If the user provides event details (event name, venue, collaborator, date, time), format them cleanly with emojis just like this:
   🌍 EVENT_NAME | Topic
   📍 Venue
   🤝 Collaborator
   📅 Date
   ⏰ Time
5. For WhatsApp captions: keep to 2 short couplets or 3 micro-stanzas with *bold* emphasis and zero #hashtags.
6. If the user asks for multiple platforms in one command (e.g., "Insta and WhatsApp both"), separate each platform block with `===SPLIT_MSG===` on its own line. If they ask for a single caption, output ONLY that ready-to-paste caption with no extra chatter or code fences.
"""

    def __init__(self, api_key: str):
        cleaned_key = (api_key or "").strip().strip('"').strip("'")
        if not cleaned_key:
            raise ValueError("Gemini API Key is required.")
        self.api_key = cleaned_key

    def _get_mime_type(self, file_path: str, default_mime: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or default_mime

    def _build_inline_part(self, file_path: str, default_mime: str) -> Dict[str, Any]:
        mime_type = self._get_mime_type(file_path, default_mime)
        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return {"inline_data": {"mime_type": mime_type, "data": encoded}}

    def _prepare_media_parts(self, file_paths: List[str], default_mime: str = "application/octet-stream") -> List[Dict[str, Any]]:
        """Send media inline if total size <= 14 MB, otherwise use Gemini Files API."""
        if not file_paths:
            return []

        total_size = sum(os.path.getsize(p) for p in file_paths)
        parts = []

        if total_size <= self.INLINE_MAX_BYTES:
            logger.info(f"Sending {len(file_paths)} file(s) inline ({total_size / (1024*1024):.2f} MB total).")
            for path in file_paths:
                parts.append(self._build_inline_part(path, default_mime))
        else:
            for path in file_paths:
                file_info = self._upload_file(path)
                parts.append({
                    "file_data": {
                        "mime_type": file_info.get("mimeType", default_mime),
                        "file_uri": file_info.get("uri")
                    }
                })
        return parts

    def _upload_file(self, file_path: str) -> Dict[str, Any]:
        """Upload any file (video or image) to Gemini File API using resumable protocol."""
        file_size = os.path.getsize(file_path)
        mime_type = self._get_mime_type(file_path, "application/octet-stream")

        display_name = os.path.basename(file_path)
        logger.info(f"Initiating Gemini upload for '{display_name}' ({file_size / (1024*1024):.2f} MB, {mime_type})...")

        headers = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(file_size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        metadata = {"file": {"display_name": display_name}}
        start_res = requests.post(
            self.UPLOAD_URL,
            headers=headers,
            json=metadata,
            timeout=30
        )
        if not start_res.ok:
            raise RuntimeError(f"Gemini upload init failed ({start_res.status_code}): {start_res.text}")

        upload_url = start_res.headers.get("X-Goog-Upload-URL")
        if not upload_url:
            raise RuntimeError("Failed to obtain Gemini resumable upload URL.")

        upload_headers = {
            "Content-Length": str(file_size),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize"
        }
        with open(file_path, "rb") as f:
            finish_res = requests.post(upload_url, headers=upload_headers, data=f, timeout=180)
            if not finish_res.ok:
                raise RuntimeError(f"Gemini file upload failed ({finish_res.status_code}): {finish_res.text}")

        file_info = finish_res.json().get("file", {})
        file_uri = file_info.get("uri")
        file_name = file_info.get("name")
        logger.info(f"Uploaded successfully. File URI: {file_uri}, Name: {file_name}")

        self._wait_for_processing(file_name)
        return file_info

    def _wait_for_processing(self, file_name: str, max_wait: int = 120):
        """Poll file state until Gemini has finished media processing."""
        logger.info("Waiting for Gemini to process uploaded media...")
        start_time = time.time()

        clean_name = file_name if not file_name.startswith("files/") else file_name[6:]
        url = f"{self.FILES_BASE_URL}/files/{clean_name}"
        headers = {"x-goog-api-key": self.api_key}

        while time.time() - start_time < max_wait:
            res = requests.get(url, headers=headers, timeout=30)
            if not res.ok:
                raise RuntimeError(f"Gemini file status check failed ({res.status_code}): {res.text}")
            data = res.json()
            state = data.get("state")
            if state == "ACTIVE":
                logger.info("Media processing complete (ACTIVE).")
                return
            elif state == "FAILED":
                raise RuntimeError("Gemini failed to process uploaded media.")
            time.sleep(3)
        logger.warning("Media processing wait timeout reached, proceeding to prompt.")

    def generate_captions(self, video_file_path: str, topic_context: str = "") -> str:
        """Analyze video and return 4 platform blocks separated by ===SPLIT_MSG===."""
        parts = self._prepare_media_parts([video_file_path], default_mime="video/mp4")

        user_content = (
            "Analyze this video carefully and generate all 4 platform blocks separated by ===SPLIT_MSG===. "
            "Keep every block short, crisp, youth-friendly, and strictly aligned with the examples (Instagram max 3 lines + hashtags). "
            "Do NOT use clichés like 'reflect the truth', 'find the truth', or 'see the truth'."
        )
        if topic_context:
            user_content += f"\nTopic / Context: {topic_context}"

        parts.append({"text": user_content})

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "system_instruction": {"parts": [{"text": self.VIDEO_SYSTEM_PROMPT}]},
            "generationConfig": {
                "temperature": 0.4,
                "topP": 0.85,
                "maxOutputTokens": 2000
            }
        }
        return self._call_gemini_models(payload)

    def generate_multi_photo_captions(self, image_file_paths: List[str], topic_context: str = "") -> str:
        """Analyze multiple photos and return 3 platform blocks separated by ===SPLIT_MSG===."""
        parts = self._prepare_media_parts(image_file_paths, default_mime="image/jpeg")

        user_content = (
            f"Analyze these {len(image_file_paths)} photos collectively as a single event/carousel and generate all 3 platform blocks separated by ===SPLIT_MSG===. "
            "Keep Instagram strictly maximum 3 lines + hashtags, youth-friendly, and crisp. "
            "Do NOT use clichés like 'reflect the truth', 'find the truth', or 'see the truth'."
        )
        if topic_context:
            user_content += f"\nTopic / Context: {topic_context}"

        parts.append({"text": user_content})

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "system_instruction": {"parts": [{"text": self.PHOTO_SYSTEM_PROMPT}]},
            "generationConfig": {
                "temperature": 0.4,
                "topP": 0.85,
                "maxOutputTokens": 1800
            }
        }
        return self._call_gemini_models(payload)

    def generate_custom_caption(
        self,
        custom_instructions: str,
        media_paths: Optional[List[str]] = None,
        topic_context: str = ""
    ) -> str:
        """Generate a custom caption from user instructions, optionally paired with video/photos."""
        parts = []
        if media_paths:
            parts.extend(self._prepare_media_parts(media_paths))

        prompt_text = f"User Instructions & Information:\n{custom_instructions.strip()}"
        if topic_context:
            prompt_text += f"\n\nAdditional Topic / Thread Context:\n{topic_context}"

        parts.append({"text": prompt_text})

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "system_instruction": {"parts": [{"text": self.CUSTOM_SYSTEM_PROMPT}]},
            "generationConfig": {
                "temperature": 0.45,
                "topP": 0.85,
                "maxOutputTokens": 2000
            }
        }
        return self._call_gemini_models(payload)

    def _call_gemini_models(self, payload: Dict[str, Any]) -> str:
        """Helper to invoke candidate Gemini models with automatic fallback."""
        last_error = None
        for model in self.CANDIDATE_MODELS:
            generate_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }
            logger.info(f"Attempting generation with model '{model}'...")

            try:
                response = requests.post(generate_url, json=payload, headers=headers, timeout=90)
                if response.status_code == 200:
                    res_json = response.json()
                    candidates = res_json.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        output_text = "".join([p.get("text", "") for p in parts]).strip()
                        logger.info(f"Successfully generated captions using '{model}'!")
                        return output_text
                elif response.status_code == 404:
                    logger.warning(f"Model '{model}' returned 404. Falling back to next model...")
                    last_error = response.text
                    continue
                else:
                    logger.error(f"Error from '{model}': {response.status_code} - {response.text}")
                    last_error = response.text
                    continue
            except Exception as req_err:
                logger.warning(f"Request failed for model '{model}': {req_err}")
                last_error = str(req_err)
                continue

        raise RuntimeError(f"All candidate Gemini models failed to generate content. Details: {last_error}")
