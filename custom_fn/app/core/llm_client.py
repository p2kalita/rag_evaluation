import os
from groq import Groq
from app.config.settings import MODEL, MAX_TOKENS

_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def _call_llm(system: str, user: str) -> str:
    response = _client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    return response.choices[0].message.content.strip()