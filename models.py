from typing import Optional
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str

class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None

class UploadRequest(BaseModel):
    source: str         # file name/label
    content: str        # file content
