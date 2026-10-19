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
    Multimodal Video Caption Generator using Google Gemini File API & generateContent.
    Incorporates title & packaging principles from Aprilynne Alter and DecodingYT,
    and outputs 4 separate messages via ===SPLIT_MSG=== delimiter.
    """
    UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"
    FILES_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    
CANDIDATE_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash-lite",
    "gemini-3-flash",
    "gemini-4-argon"
]


    SYSTEM_PROMPT = """
You are an elite viral content strategist and philosophical editor (specializing in Bhagavad Gita, Vedanta, self-inquiry, and practical life wisdom).
Watch and listen to the provided video carefully. Match the primary language/tone of the video (conversational Hindi/Hinglish with clean Latin English words, or English if spoken in English).

Your goal is to generate FOUR separate platform outputs.
CRITICAL RULE: Separate each of the 4 outputs with the exact delimiter:
===SPLIT_MSG===
Do NOT add any extra intro or outro text outside the 4 blocks.

Follow these strict rules for each block:

------------------------------------------------------------
BLOCK 1: WHATSAPP POST
------------------------------------------------------------
📱 *WHATSAPP POST*

• Hook (Line 1): Open a strong Curiosity Loop (an unfinished thought, counter-intuitive observation, or sharp human contradiction from the video). NEVER declare or preach the final conclusion in the first line.
• Body (2–3 short paragraphs): Deconstruct the surface belief/excuse and reveal the deeper root cause spoken about in the video.
• Language & Formatting:
  - Write in simple, conversational Hindi (Devanagari) mixed naturally with familiar single English words in Latin script (e.g., Lifestyle, Symptom, Consumption, Mindset, Trap). Never use dual-bracketed translations or heavy Sanskritized jargon.
  - Use WhatsApp syntax: *bold* for core concepts/contrasts, and _italics_ for subtle inner thoughts.
  - Place 1 relevant emoji at the start of each paragraph (never cluster emojis side-by-side).
• Closing: End with 1 piercing introspective question that forces the reader to reflect all day.

===SPLIT_MSG===

------------------------------------------------------------
BLOCK 2: WHATSAPP STATUS
------------------------------------------------------------
💬 *WHATSAPP STATUS*

• Write 1 to 2 ultra-punchy, high-impact lines (strictly under 20 words total).
• Must have high intake speed and open an unresolved mental loop that makes viewers pause on the status or watch the full clip.

===SPLIT_MSG===

------------------------------------------------------------
BLOCK 3: INSTAGRAM REEL CAPTION
------------------------------------------------------------
📸 *INSTAGRAM*

• Line 1 (Scroll-Stopping Hook): Strictly under 8 words. Front-load a Curiosity or Emotional Power Word so the viewer is hooked before the "...more" cutoff.
• Body: 3–4 crisp, value-packed lines summarizing the core shift in perspective from the video.
• Call to Action (CTA): 1 natural line prompting a Save or Share (e.g., "Save this reminder, and share it with someone who needs this clarity today.").
• Hashtags: 12 to 15 targeted hashtags mixing niche philosophy and high-reach personal growth tags.

===SPLIT_MSG===

------------------------------------------------------------
BLOCK 4: YOUTUBE SHORTS (Aprilynne Alter + DecodingYT Framework)
------------------------------------------------------------
▶️ *YOUTUBE SHORTS*

🎯 *Title Options (Write for Humans & CTR, NOT Keyword Stuffing):*
Provide 3 distinct title options following these exact rules:
1. Length & Intake Speed: Strictly UNDER 55 characters so it reads in 1 second and never truncates on mobile.
2. Front-Load Core Words: Put the most important subject/hook words in the first half of the title.
3. Open a Curiosity Loop: Make the viewer's brain ask "Why?" or "How?" (never write a boring descriptive label).
4. Power Word in ALL CAPS: Capitalize exactly 1 emotional, urgency, authority, or curiosity power word in ALL CAPS (e.g., TRUTH, TRAP, STOP, SECRET, MISTAKE, REAL, HIDDEN, NEVER).

📝 *Description (Write for YouTube Search / SEO):*
• Top 2–3 Lines (Natural Keyword Overview): Write a smooth 2–3 sentence overview of the video that naturally embeds the main search keywords without sounding robotic.
• Search Queries (Queries Solved): List 5–6 realistic search phrases that viewers actually type into YouTube search to find this topic (include both Hinglish/Hindi in Latin script and English phrases).
• Hashtags: 3–5 relevant hashtags including #Shorts.

🏷️ *Tags (Copy-Paste Ready):*
Provide the 6–8 core search keyword phrases as a single comma-separated line ready to paste into the YouTube Studio tags box.
"""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Gemini API Key is required.")
        self.api_key = api_key

    def _upload_video(self, file_path: str) -> Dict[str, Any]:
        """Upload video file to Gemini File API using resumable protocol."""
        file_size = os.path.getsize(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "video/mp4"

        display_name = os.path.basename(file_path)
        logger.info(f"Initiating Gemini upload for '{display_name}' ({file_size / (1024*1024):.2f} MB)...")

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
            f"{self.UPLOAD_URL}?key={self.api_key}",
            headers=headers,
            json=metadata,
            timeout=30
        )
        start_res.raise_for_status()

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
            finish_res.raise_for_status()

        file_info = finish_res.json().get("file", {})
        file_uri = file_info.get("uri")
        file_name = file_info.get("name")
        logger.info(f"Uploaded successfully. File URI: {file_uri}, Name: {file_name}")

        self._wait_for_processing(file_name)
        return file_info

    def _wait_for_processing(self, file_name: str, max_wait: int = 120):
        """Poll file state until Gemini has finished audio/video transcoding."""
        logger.info("Waiting for Gemini to process video frames and audio...")
        start_time = time.time()
        
        clean_name = file_name if not file_name.startswith("files/") else file_name[6:]
        url = f"{self.FILES_BASE_URL}/files/{clean_name}?key={self.api_key}"
        headers = {"x-goog-api-key": self.api_key}

        while time.time() - start_time < max_wait:
            res = requests.get(url, headers=headers, timeout=30)
            res.raise_for_status()
            data = res.json()
            state = data.get("state")
            if state == "ACTIVE":
                logger.info("Video processing complete (ACTIVE).")
                return
            elif state == "FAILED":
                raise RuntimeError("Gemini failed to process uploaded video.")
            time.sleep(4)
        logger.warning("Video processing wait timeout reached, proceeding to prompt.")

    def generate_captions(self, video_file_path: str, topic_context: str = "") -> List[str]:
        """Upload video, invoke Gemini, and return a list of 4 separate platform caption strings."""
        file_info = self._upload_video(video_file_path)
        file_uri = file_info.get("uri")
        mime_type = file_info.get("mimeType", "video/mp4")

        user_content = "Analyze this video and generate all 4 platform blocks separated by ===SPLIT_MSG===."
        if topic_context:
            user_content += f"\nAdditional Topic Context: {topic_context}"

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"file_data": {"mime_type": mime_type, "file_uri": file_uri}},
                        {"text": user_content}
                    ]
                }
            ],
            "system_instruction": {
                "parts": [{"text": self.SYSTEM_PROMPT}]
            },
            "generationConfig": {
                "temperature": 0.45,
                "topP": 0.9,
                "maxOutputTokens": 2500
            }
        }

        last_error = None
        for model in self.CANDIDATE_MODELS:
            generate_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }
            logger.info(f"Attempting caption generation with model '{model}'...")
            
            try:
                response = requests.post(generate_url, json=payload, headers=headers, timeout=90)
                if response.status_code == 200:
                    res_json = response.json()
                    candidates = res_json.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        output_text = "".join([p.get("text", "") for p in parts]).strip()
                        logger.info(f"Successfully generated captions using '{model}'!")
                        
                        blocks = [b.strip() for b in output_text.split("===SPLIT_MSG===") if b.strip()]
                        return blocks if blocks else [output_text]
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
