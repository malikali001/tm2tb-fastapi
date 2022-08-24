import os

from dotenv import load_dotenv
from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKey, APIKeyHeader, APIKeyQuery
from starlette.status import HTTP_403_FORBIDDEN

from database import SessionLocal
from jwt_authentication import JwtAuthentication

load_dotenv()
API_KEY = os.environ.get("API_KEY")
API_KEY_NAME = os.environ.get("API_KEY_NAME")
JWT_SECRET = os.environ.get("BOOTHMATE_AUTH_TOKEN_SECRET")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


api_key_query = APIKeyQuery(name=API_KEY_NAME, auto_error=False)
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(
    api_key_query: str = Security(api_key_query),
    api_key_header: str = Security(api_key_header),
):
    if api_key_query == API_KEY:
        return api_key_query
    elif api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
        )
