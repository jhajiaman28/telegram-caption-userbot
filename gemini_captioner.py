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
    Features automatic model fallback (gemini-2.0-flash, gemini-1.5-flash-latest, etc.)
    and resilient error handling.
    """
    UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"
    FILES_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    
    CANDIDATE_MODELS = [
        "gemini-3.8-flash",
        "gemini-4-argon",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
        "gemini-3.1-pro",
        "gemini-1.5-flash"
    ]

    SYSTEM_PROMPT = """
You are an expert social media strategist and content editor specializing in spiritual, philosophical (Bhagavad Gita, Vedanta), and wisdom-based short-form video content.
You will be provided with a short video (or Reel/Short).
Your task is to watch/listen to the video carefully and produce FOUR distinct, ready-to-publish captions in Hindi / Hinglish / English (matching the primary language spoken in the video).

FORMAT YOUR ENTIRE OUTPUT EXACTLY AS FOLLOWS WITH THESE FOUR LABELS:

📱 *1. WHATSAPP POST*
• Engaging first-line hook that sparks deep curiosity without giving away the full lesson upfront.
• 2-3 short, powerful paragraphs capturing the core insight, breaking common misconceptions, and exploring human desires/habits.
• Bold critical concepts (*bold*), italicize introspective thoughts (_italics_).
• Avoid heavy esoteric jargon; keep English words in clean Latin script (e.g. Lifestyle, Symptom, Consumption).
• End with an introspective reflection question.

💬 *2. WHATSAPP STATUS*
• Exactly 1 to 3 punchy, high-impact lines crafted specifically for mobile screen reading.
• Direct, thought-provoking, and curiosity-inducing.

📸 *3. INSTAGRAM*
• Line 1: Strong scroll-stopping hook (under 10 words).
• Body: 3-5 lines of concise value-packed summary delivering the video's core wisdom.
• Call to Action (CTA): e.g. "Share this with someone who needs this clarity today." or "Save for later."
• Hashtags: 15 to 20 highly relevant, targeted hashtags (e.g. #BhagavadGita #SelfAwareness #Mindset #Clarity #Wisdom #LifeLessons #Vedanta).

▶️ *4. YOUTUBE SHORTS*
• Title: Click-worthy, high-CTR Title (strictly under 90 characters).
• Description: 2-3 concise sentences summarizing the key takeaway.
• Tags / Hashtags: #Shorts #Wisdom #ShortVideo #LifeLessons + 2 topic-specific tags.

Ensure the wisdom is accurately grounded in what is actually spoken in the video.
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
        """Upload video, invoke Gemini, and return the 4 platform captions with auto-fallback."""
        file_info = self._upload_video(video_file_path)
        file_uri = file_info.get("uri")
        mime_type = file_info.get("mimeType", "video/mp4")

        user_content = "Analyze this video and generate all 4 platform captions."
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
                "maxOutputTokens": 2048
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
