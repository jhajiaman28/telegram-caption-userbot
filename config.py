import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class UserbotConfig:
    api_id: int
    api_hash: str
    gemini_api_key: str
    session_string: Optional[str] = None
    session_name: str = "caption_userbot"
    trigger_keyword: str = "[final]"
    download_dir: str = "downloads"

    @classmethod
    def from_env(cls) -> "UserbotConfig":
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        api_id_raw = os.getenv("TELEGRAM_API_ID", "").strip()
        api_hash = os.getenv("TELEGRAM_API_HASH", "").strip()
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        session_str = os.getenv("TELEGRAM_SESSION_STRING", "").strip() or None
        session_name = os.getenv("SESSION_NAME", "caption_userbot").strip()
        trigger_keyword = os.getenv("TRIGGER_KEYWORD", "[final]").strip().lower()
        download_dir = os.getenv("DOWNLOAD_DIR", "downloads").strip()

        api_id = int(api_id_raw) if api_id_raw.isdigit() else 0

        return cls(
            api_id=api_id,
            api_hash=api_hash,
            gemini_api_key=gemini_key,
            session_string=session_str,
            session_name=session_name,
            trigger_keyword=trigger_keyword,
            download_dir=download_dir
        )
