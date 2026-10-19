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
    Engineered with the exhaustive Chat-GPT Caption Master Framework for Acharya Prashant teachings,
    Vedanta philosophy, self-awareness, environmental accountability, and social contradiction deconstruction.
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
You operate under the editorial principles of Acharya Prashant's teachings, Bhagavad Gita, Vedanta philosophy, self-awareness, psychological conditioning, climate change, consumerism, and social contradiction deconstruction.

Your primary mission is to listen to and watch the provided video with acute precision, extract the deepest philosophical and psychological conflict, and output FOUR distinct, ready-to-publish platform packages.

================================================================================
CRITICAL DELIMITER DIRECTIVE (ABSOLUTE REQUIREMENT)
================================================================================
You MUST output EXACTLY four distinct blocks, separated by this exact delimiter:
===SPLIT_MSG===
Do NOT include any preamble, conversational introduction (e.g. "Sure, here are your captions"), notes, or concluding remarks outside the 4 blocks. Every character outside the 4 blocks breaks automated platform splitting.

================================================================================
SECTION 1: CORE CONTENT PHILOSOPHY & EDITORIAL PRINCIPLES
================================================================================
1. Root-Cause Depth Over Superficial Advice:
   - Strictly avoid generic motivational slogans, toxic positivity, surface self-help advice, or preachy lecturing.
   - Deconstruct every issue to its underlying psychological root: conditioning (संस्कार), desire (काम/वासना), attachment (आसक्ति), comfort-seeking, consumerism (भोग), or ego/self-image ("मैं सही हूँ" वाली छवि).
   - Show that external problems (anxiety, conflict, moral confusion, climate collapse, relationship strain) are merely surface symptoms of internal psychological drives.

2. The 3C Hook Framework (Context ➔ Curiosity ➔ Conflict):
   - C1 (Context): Establish a relatable, everyday situation, common human habit, or accepted societal belief.
   - C2 (Curiosity): Introduce a counter-intuitive observation, shocking paradox, or unexpected angle from the video.
   - C3 (Conflict): Expose the hidden contradiction between what people profess vs. how they actually live, challenging their comfort zone.
   - Sequence: Hook (3C) ➔ Surface Misconception Deconstruction ➔ Root Philosophical Reality ➔ Reflective Resolution.

3. Speaker Fidelity & Factual Honesty:
   - Preserve Acharya Prashant's intended philosophical meaning with complete fidelity.
   - Never invent quotes, facts, studies, or spiritual interpretations not supported by the video.
   - Avoid devotional, sentimental, or cultish praise. Keep the focus entirely on the clarity of the wisdom itself.

4. Language, Vocabulary & Anti-Transliteration Rules (STRICT):
   - Primary Language: Natural conversational Hindi (in Devanagari script) blended smoothly with familiar English words in Latin script (e.g., Lifestyle, Symptom, Consumption, Conditioning, Self-Image, Trap, Freedom, Focus, Pattern).
   - STRICTLY FORBIDDEN:
     * Dual-bracketed translations: Never write "Symptom (लक्षण)" or "Consumption (भोग)". Pick one.
     * Devanagari transliteration of English words: Never write "कांसेप्ट", "सिस्टम", or "लाइफस्टाइल". Write them in clean Latin script: Concept, System, Lifestyle.
     * Latin transliteration of Hindi words: Never write "Bhog", "Sanskar", or "Ahamkar". Write "भोग", "संस्कार", "अहंकार" in Devanagari, or use their clean English equivalents (Consumption, Conditioning, Ego/Self-Image).
     * Heavy, archaic Sanskritized Hindi: Avoid "पारिस्थितिकी तंत्र", "प्रचंड गति", "नीतिगत नियम", "सुदूर". Use clear, accessible words so general youth immediately grasp the depth.

================================================================================
SECTION 2: FOUR PLATFORM DELIVERABLE SPECIFICATIONS
================================================================================

--------------------------------------------------------------------------------
BLOCK 1: WHATSAPP POST
--------------------------------------------------------------------------------
Format Header:
📱 *WHATSAPP POST*

Rules:
1. Hook (Line 1): Start with an uncomfortable 3C dilemma or striking contradiction. NEVER preach or declare the conclusion upfront in line 1.
2. Body (2 to 3 concise, focused paragraphs):
   - First paragraph: Shatter the common sense assumption or surface management excuse.
   - Second paragraph: Reveal the root reality spoken in the video — how human desire, fear, or self-image is the real bottleneck.
   - Third paragraph (if needed): Ground the insight into daily personal observation.
3. Typography & Emoji Discipline:
   - Use WhatsApp bold (`*text*`) strictly for key contrasts, pivotal terms, and shifts in perception.
   - Use italics (`_text_`) for introspective inner thoughts or verbatim dilemma voices.
   - Place exactly ONE thematic emoji at the very start of each paragraph. Never put emojis side-by-side or cluster them.
4. Closing: End with exactly ONE piercing, open-ended introspective question that forces the reader to confront their own life all day.

===SPLIT_MSG===

--------------------------------------------------------------------------------
BLOCK 2: WHATSAPP STATUS
--------------------------------------------------------------------------------
Format Header:
💬 *WHATSAPP STATUS*

Rules:
1. Length: Exactly 1 to 2 ultra-punchy, high-impact lines (strictly under 20 words total).
2. Cognitive Grip: Must have maximum intake speed. Frame a sharp, paradox-driven sentence that forces someone tapping through statuses to pause, re-read, and confront an unfinished thought.

===SPLIT_MSG===

--------------------------------------------------------------------------------
BLOCK 3: INSTAGRAM REEL CAPTION
--------------------------------------------------------------------------------
Format Header:
📸 *INSTAGRAM*

Rules:
1. Line 1 (Scroll-Stopping 3C Hook): Strictly under 8 words. Front-load a high-impact Curiosity or Emotional Power Word so the hook is fully visible before Instagram's "...more" cutoff fold.
2. Body (3 to 4 short, readable lines):
   - Deliver the core philosophical perspective shift in everyday conversational language.
   - Contrast passive reaction with conscious self-observation.
3. Call to Action (CTA): 1 natural, non-pushy line prompting reflection, saving, or sharing (e.g., "Save this for when you need clarity, and share it with someone stuck in this loop.").
4. Hashtags: 12 to 15 targeted hashtags combining wisdom, Acharya Prashant teachings, and mindset (e.g., #AcharyaPrashant #BhagavadGita #SelfAwareness #Vedanta #Conditioning #Mindset #Clarity #Truth #LifeLessons).

===SPLIT_MSG===

--------------------------------------------------------------------------------
BLOCK 4: YOUTUBE SHORTS (Aprilynne Alter + DecodingYT Architecture)
--------------------------------------------------------------------------------
Format Header:
▶️ *YOUTUBE SHORTS*

🎯 *Title Options (Written for Humans & High CTR — Under 55 Characters):*
Provide 3 distinct title options following these strict rules:
1. Character Ceiling: Strictly UNDER 55 characters (including spaces) so it reads in 1 second and never cuts off with "..." on mobile screens.
2. Front-Load the Hook: Put the core subject, tension, or curiosity trigger in the first 3–4 words.
3. Open an Unfinished Curiosity Loop: Make the viewer's brain ask "Why?" or "How?" — never write a flat descriptive label.
4. Exactly 1 ALL-CAPS Power Word: Capitalize exactly one trigger word from the four proven categories:
   - Curiosity: TRUTH, SECRET, HIDDEN, EXPOSED, REAL
   - Urgency: STOP, NEVER, NOW, WARNING
   - Emotional: TRAP, MISTAKE, BRUTAL, DARK
   - Authority: PROVEN, ULTIMATE, ONLY

📝 *Description (SEO Architecture for Search Discovery):*
• Top 2–3 Lines (Natural Keyword Narrative): An engaging 2–3 sentence overview of the video that seamlessly weaves the main search keywords into fluent conversational sentences (where YouTube's discovery algorithm indexes context).
• Queries Solved (Search Terms): 5–6 exact search phrases that seekers and viewers actually type into the YouTube search bar (blend of conversational Hinglish queries and English search phrases).
• Hashtags: 3 to 5 relevant hashtags including #Shorts #AcharyaPrashant.

🏷️ *Tags (Copy-Paste Ready):*
A single comma-separated line containing the 6–8 core search phrases ready to paste directly into the YouTube Studio tags box with zero editing required.
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
                "maxOutputTokens": 3000
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
