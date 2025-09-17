import os
from dotenv import load_dotenv  # type: ignore

load_dotenv()

# OpenRouter configuration (primary - FREE)
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
API_KEY = OPENROUTER_API_KEY
