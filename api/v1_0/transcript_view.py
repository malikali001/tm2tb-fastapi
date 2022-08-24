from dependencies import APIKey, get_api_key, get_db
from fastapi import APIRouter, Depends
from models import Transcript
from pydantic import BaseModel
from sqlalchemy.orm import Session


class RequestDataForTranscript(BaseModel):
    """
    Request data model.

    It represents the source and target lang

    meeting_id : strings
        strings representing the meeting_id
    lang_code : strings
        strings representing the lang code
    interpreter_audio : boolean
        boolean representing interpreter true or false
    text : strings
        strings representing the text
    """

    meeting_id: str
    lang_code: str
    interpreter_audio: bool
    text: str


router = APIRouter()


@router.post("/transcript-insert")
async def transcript(
    data: RequestDataForTranscript,
    db: Session = Depends(get_db),
):
    Transcript.create(db, data)
    return {"message": "Transcript successfully inserted"}
