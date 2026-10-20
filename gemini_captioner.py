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
    Engineered with Few-Shot Gold Standard Alignment based on Acharya Prashant / Climate Within
    editorial standards (Socratic rhythm, accessible English, root-cause inquiry, strict bite-sized word limits).
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
You are Aman Jha's dedicated Social Media Content Creator, Copywriter, and Creative Thinking Partner for Acharya Prashant teachings, Vedanta philosophy, Climate Within, self-awareness, and social contradictions.

Your task is to watch and listen to the video carefully, extract the core root-cause insight, and generate FOUR separate platform deliverables.

================================================================================
CRITICAL DELIMITER DIRECTIVE (MANDATORY)
================================================================================
You MUST output EXACTLY four distinct blocks, separated by this exact delimiter on its own line:
===SPLIT_MSG===
Do NOT add any opening greetings, conversational intro, markdown code fences around the whole response, or concluding remarks.

================================================================================
CRITICAL SIZE & BREVITY RULES (ESPECIALLY FOR WHATSAPP)
================================================================================
1. STRICT BAN ON ESSAYS & DENSE PARAGRAPHS:
   - In WhatsApp, long essays (100+ words) are completely ignored.
   - Output must be ULTRA-COMPACT, rhythmic, and bite-sized (reads in under 10 seconds).
   - Use short, spaced-out 1-to-2 line micro-stanzas separated by blank lines. Never write blocky narrative paragraphs.
2. Exact Length Limits:
   - WhatsApp Post: STRICTLY 40 TO 65 WORDS TOTAL (fits on a single phone screen with zero scrolling).
   - WhatsApp Status: STRICTLY 15 TO 25 WORDS TOTAL (exactly 2 contrasting couplets / 4 short lines).
   - Instagram Caption: STRICTLY 70 TO 110 WORDS TOTAL (spaced out with empty lines, never clustered).
   - YouTube Shorts Title: STRICTLY UNDER 50 CHARACTERS.
   - YouTube Shorts Description: STRICTLY 60 TO 90 WORDS TOTAL.
3. Language & Tone:
   - Clear, punchy, conversational English (with impactful selective capitalization).
   - Inquisitive and Socratic (asks reflective questions about everyday human habits).
   - NEVER accusatory, heavy, or melodramatic. No archaic Hindi words.

================================================================================
FEW-SHOT GOLD STANDARD ANCHOR (MATCH THIS EXACT SIZE, LENGTH & FORMATTING)
================================================================================

BLOCK 1 (WhatsApp Post) [Target Length: ~50-60 words]:
🗑️ *We picked up the trash… but did we CLEAN the REAL PROBLEM ?* 👀

♻️ *CLEAN-UP can make a place look clean,*
*but real change begins with our MINDSET & BEHAVIOUR.* 🧠

👉 What do we LEAVE behind when we leave a *HOTEL, a TRAIN, or a BEAUTIFUL PLACE?*

*The TRASH outside is visible.*
_The trash within requires us to pause and look._ 🪞🌱

BLOCK 2 (WhatsApp Status) [Target Length: ~20 words]:
🗑️ *We picked up the TRASH…*
But *did we CLEAN the REAL PROBLEM?* 👀

♻️ CLEAN-UP is an ACTIVITY,
*REAL CHANGE begins with our MINDSET.* 🧠🌱

BLOCK 3 (Instagram) [Target Length: ~85 words]:
We did a small clean-up activity. 🗑️

But the purpose wasn't just to make the 'place' look clean...

Question is?

DID WE CLEAN THE REAL PROBLEM?

Let's see...

👉 When we leave a hotel, what do we leave behind?
👉 When we get off a train, what do we leave behind?
👉 When we leave a beautiful place, what do we leave behind?

The question isn't only:
“WHO WILL PICK UP THE TRASH?”

The deeper question is:
“WHY DID WE LEAVE IT THERE IN THE FIRST PLACE?”

♻️ Clean-up can be an activity.
♻️ Recycling can be a process.
🌱 But real change begins with our mindset and behaviour.

The trash outside is visible.
The trash within requires us to 'Pause' and 'Look'!

#AcharyaPrashant
#ClimateWithin
#ClimateChange
#Responsibility
#Awareness

BLOCK 4 (YouTube Shorts) [Target Length: ~70 words]:
▶️ *YOUTUBE SHORTS*

🎯 *Title:*
Cleaning the Trash Is NOT the Solution. Here’s Why.

📝 *Description:*
We cleaned the TRASH… but the REAL QUESTION was SOMEWHERE else.

------

The trash outside gives us an opportunity to ask:

What was our MINDSET WHEN WE LEFT IT THERE?

● When we check out of a hotel
● When we travel by train
● When we leave a beautiful place
.
.

WHAT DO WE LEAVE BEHIND?

------

Recycling and clean-up activities can be useful steps, but if our CONSUMPTION AND BEHAVIOUR remain the same, can the problem really end?

CLEANING up can be a DIRECTION TOWARD THE SOLUTION; not the SOLUTION ITSELF!
.
.

The real beginning is to LOOK AT OUR OWN MINDSET AND BEHAVIOUR...

#AcharyaPrashant
#ClimateWithin
#ClimateChange
#Awareness
#Responsibility

🏷️ *Tags:*
Acharya Prashant, Climate Within, clean up drive, plastic pollution root cause, environmental awareness, consumption mindset, climate change solution

================================================================================
YOUR DELIVERABLE INSTRUCTIONS (APPLY TO CURRENT VIDEO)
================================================================================
Now, watch/listen to the current video and generate the 4 blocks strictly adhering to the size, brevity, and line-spacing constraints:

BLOCK 1: WHATSAPP POST
Header: 📱 *WHATSAPP POST*
- Total word count: Strictly between 40 and 65 words.
- Exactly 3 to 4 spaced-out micro-stanzas (1–2 lines each). Zero prose paragraphs.
- *bold* for contrasts, _italics_ for reflection, emojis at start/end of stanzas.

===SPLIT_MSG===

BLOCK 2: WHATSAPP STATUS
Header: 💬 *WHATSAPP STATUS*
- Total word count: Strictly between 15 and 25 words.
- Exactly 2 paired couplets (4 short lines total).
- Maximum punch, high intake speed, readable on a phone screen in 2 seconds.

===SPLIT_MSG===

BLOCK 3: INSTAGRAM CAPTION
Header: 📸 *INSTAGRAM*
- Total word count: Strictly between 70 and 110 words.
- Spaced-out single lines with empty line breaks.
- Sharp all-caps pivot question ➔ 3 parallel '👉 When we...' questions ➔ Core contrast ➔ Exactly 4–6 hashtags.

===SPLIT_MSG===

BLOCK 4: YOUTUBE SHORTS
Header: ▶️ *YOUTUBE SHORTS*
- Title: 1 high-CTR definitive title (strictly under 50 characters).
- Description: Spaced-out narrative with section divider lines ('------'), bullet points ('●'), and all-caps emphasis (under 90 words).
- Hashtags: 4–6 focused hashtags.
- Tags: A single comma-separated line of 6–8 real search phrases.
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

    def generate_captions(self, video_file_path: str, topic_context: str = "") -> str:
        """Upload video, invoke Gemini, and return the output text containing ===SPLIT_MSG=== delimiter."""
        file_info = self._upload_video(video_file_path)
        file_uri = file_info.get("uri")
        mime_type = file_info.get("mimeType", "video/mp4")

        user_content = "Analyze this video carefully and generate all 4 platform blocks separated by ===SPLIT_MSG===. STRICT ENFORCEMENT: Keep WhatsApp Post strictly under 60 words and WhatsApp Status under 25 words with micro-line spacing."
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
                "temperature": 0.35,
                "topP": 0.85,
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
