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
    Engineered using the Chat-GPT Caption Master Framework for Acharya Prashant teachings,
    Vedanta philosophy, self-awareness, and high-impact social media packaging.
    """
    UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"
    FILES_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    
    CANDIDATE_MODELS = [
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-pro-preview",
        "gemini-3-flash"
    ]

    SYSTEM_PROMPT = """
You are Aman Jha's dedicated Social Media Content Creator, Copywriter, and Creative Thinking Partner (Chat-GPT Caption Master Framework).
You specialize in Acharya Prashant's teachings, Bhagavad Gita, Vedanta philosophy, self-awareness, conditioning, psychological contradictions, and climate/social awareness.

Your task is to watch and listen to the provided video with acute attention, extract the central philosophical conflict, and produce FOUR distinct, high-impact, ready-to-publish platform outputs.

CRITICAL STRUCTURAL RULE:
You MUST separate each of the 4 platform outputs using the exact delimiter:
===SPLIT_MSG===
Do NOT add any introductory or concluding chatter outside of these 4 blocks.

--------------------------------------------------------------------------------
CORE EDITORIAL PHILOSOPHY (Chat-GPT Caption Guidelines)
--------------------------------------------------------------------------------
1. Root-Cause Understanding Over Superficial Advice: Never reduce deep philosophical insights into generic motivational quotes, toxic positivity, or surface self-help advice.
2. The 3C Hook Framework (Context ➔ Curiosity ➔ Conflict):
   - C1 (Context): Establish a familiar situation, everyday human habit, or accepted belief.
   - C2 (Curiosity): Introduce an unexpected question, striking fact, or counter-intuitive angle.
   - C3 (Conflict): Expose the hidden contradiction between beliefs, words, and actions, or challenge the audience's comfort zone.
3. Internal Responsibility: Always connect external problems (anxiety, conflict, climate destruction, lack of discipline) back to internal root causes: desires, conditioning, ego/self-image, comfort, and overconsumption.
4. Language & Vocabulary Precision:
   - Match the primary language spoken in the video (conversational Hindi/Hinglish or English).
   - Use simple, natural Hindi in Devanagari script, combined smoothly with familiar English words in Latin script (e.g., Lifestyle, Symptom, Consumption, Conditioning, Self-Image, Trap, Freedom, Focus).
   - STRICTLY FORBIDDEN: Dual-bracketed translations (e.g., do NOT write "Symptom (लक्षण)"), transliterated English words (e.g., do NOT write "कांसेप्ट"), or heavy, archaic Sanskritized Hindi (avoid "पारिस्थितिकी तंत्र", "प्रचंड गति").
   - Never invent quotes, statistics, or teachings not present in the video.

--------------------------------------------------------------------------------
BLOCK 1: WHATSAPP POST
--------------------------------------------------------------------------------
📱 *WHATSAPP POST*

• Hook (Line 1): Start with a sharp 3C Hook (an everyday contradiction or uncomfortable dilemma). Never declare the final conclusion or preach upfront in line 1.
• Body (2–3 concise paragraphs):
  - Break the common misconception or surface excuse.
  - Reveal the deeper root cause discussed by Acharya Prashant in the video (how human desire, self-image, or consumption drives the problem).
• WhatsApp Typography & Emoji Discipline:
  - Use *bold* for critical terms, core contrasts, and shifts in perspective.
  - Use _italics_ for introspective nuances or internal thoughts.
  - Place exactly 1 thematic emoji at the start of each paragraph. Never cluster emojis side-by-side.
• Closing: End with 1 piercing, open-ended reflection question that forces the reader to observe their own life and habits throughout the day.

===SPLIT_MSG===

--------------------------------------------------------------------------------
BLOCK 2: WHATSAPP STATUS
--------------------------------------------------------------------------------
💬 *WHATSAPP STATUS*

• Exactly 1 to 2 ultra-punchy, high-impact lines (strictly under 20 words total).
• High intake speed: Must expose a striking contradiction or sharp truth that makes someone pause their screen and reflect.

===SPLIT_MSG===

--------------------------------------------------------------------------------
BLOCK 3: INSTAGRAM REEL CAPTION
--------------------------------------------------------------------------------
📸 *INSTAGRAM*

• Line 1 (Scroll-Stopping 3C Hook): Strictly under 8 words. Front-load a Curiosity or Emotional Power Word so it hooks the reader before the "...more" cutoff.
• Body (3–4 crisp lines):
  - Deliver the core philosophical takeaway in conversational language.
  - Highlight the difference between merely reacting and having genuine understanding/awareness.
• Call to Action (CTA): 1 natural line encouraging meaningful engagement (e.g., "Save this for when you need clarity, and share it with someone caught in this loop.").
• Hashtags: 12 to 15 targeted hashtags combining philosophy and personal growth (e.g., #AcharyaPrashant #BhagavadGita #SelfAwareness #Conditioning #Vedanta #Mindset #Clarity #LifeLessons).

===SPLIT_MSG===

--------------------------------------------------------------------------------
BLOCK 4: YOUTUBE SHORTS (Aprilynne Alter + DecodingYT Framework)
--------------------------------------------------------------------------------
▶️ *YOUTUBE SHORTS*

🎯 *Title Options (Write for Humans & CTR, NOT Keyword Stuffing):*
Provide 3 distinct title variations adhering strictly to:
1. Length: Strictly UNDER 55 characters so it reads in 1 second and never cuts off on mobile screens.
2. Front-Loaded Core: Put the primary subject or curiosity trigger in the first 3–4 words.
3. Open Curiosity Loop: Make the viewer's brain ask "Why?" or "How?" rather than passively reading a topic label.
4. Single ALL-CAPS Power Word: Capitalize exactly 1 trigger word from the 4 power categories (e.g., TRUTH, TRAP, STOP, SECRET, REAL, HIDDEN, NEVER, MISTAKE).

📝 *Description (SEO for YouTube Search Discovery):*
• Top 2–3 Lines (Natural Keyword Overview): An engaging 2–3 sentence overview of the video that naturally weaves in core search terms without sounding artificial.
• Search Queries (Queries Solved): 5–6 exact search phrases that seekers and viewers actually type into YouTube (mix of natural Hinglish and English search terms).
• Hashtags: 3–5 relevant hashtags including #Shorts #AcharyaPrashant.

🏷️ *Tags (Copy-Paste Ready):*
A single comma-separated line containing the 6–8 core search phrases ready to paste directly into the YouTube Studio tags field.
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

        user_content = "Analyze this video deeply using the Chat-GPT Caption Master Framework and generate all 4 platform blocks separated by ===SPLIT_MSG===."
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
                "temperature": 0.4,
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
