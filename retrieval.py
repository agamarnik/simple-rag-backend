def extract_text(response) -> str:
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""  # fallback, shouldn't normally happen

def rewrite_query(client, question: str, history: list) -> str:
    if not history:
        return question     # no history yet, nothing to resolve

    # Build a compact transcript of recent turns for context
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-6:]])

    rewrite_prompt = f"""Given this conversation history and a follow-up question, rewrite the follow-up question to be fully self-contained (resolve any pronouns or implicit references), without changing its meaning. Return ONLY the rewritten question, nothing else.

Conversation history:
{history_text}

Follow-up question: {question}

Rewritten question:"""

    try:
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=100,
            messages=[{"role": "user", "content": rewrite_prompt}]
        )
        return extract_text(response).strip()
    except Exception:
        # If rewriting fails, fall back to the original question rather than crashing
        return question
