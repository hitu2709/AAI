import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "openai/gpt-oss-120b"

TOP_K = 3
MAX_RETRIES = 2

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50