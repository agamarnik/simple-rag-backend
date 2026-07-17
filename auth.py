import os
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    expected_key = os.getenv("API_SECRET_KEY")
    if api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

    return api_key
