from typing import List, Optional

from dependencies import APIKey, JwtAuthentication, get_api_key, get_db
from fastapi import APIRouter, Depends, HTTPException, Request
from helpers import check_timestamp, generate_and_save_summary
from models import Summary
from pydantic import BaseModel
from sqlalchemy.orm import Session
from tm2tb import Summarizer


class RequestDataForSummary(BaseModel):
    """
    Request data model.

    It represents the source and target texts and their language identifiers.

    texts : List of strings
        List of strings representing the texts or sentences.
    lang : string
        Optional language identifier
    summary_sentences_n : int
        Optional number of summary sentences to return
    """

    texts: List[str]
    lang: Optional[str] = None
    summary_sentences_n: Optional[int] = 15


class RequestDataForModifySummary(BaseModel):
    """Request data model.

    It represents the source lang

    lang : strings
        strings representing the lang code.
    """

    lang: str


class ResponseSummaryData(BaseModel):
    """
    Response summary data model
    It represents the source language summary

    summary_text :
        String representing the summary
    """

    summary: List[str]


router = APIRouter()


@router.post("/summary", response_model=ResponseSummaryData)
async def summarize(
    data: RequestDataForSummary, api_key: APIKey = Depends(get_api_key)
):
    summarizer = Summarizer(
        texts=data.texts, lang=data.lang, summary_sentences_n=data.summary_sentences_n
    )
    summary = summarizer.extract_summary()
    return {"summary": summary}


@router.post("/meetings/{meeting_id}/summary", response_model=ResponseSummaryData)
async def summary(
    data: RequestDataForModifySummary,
    meeting_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    access_token = request.headers.get("access_token")
    if JwtAuthentication.secure(access_token, meeting_id):
        output = []
        summary = Summary.fetch(db, meeting_id, data.lang)
        if summary:
            if check_timestamp(db, summary):
                output = summary.text.split(" | ")
            else:
                output = generate_and_save_summary(
                    db, meeting_id, data.lang, summary_id=summary.id
                )
        else:
            output = generate_and_save_summary(db, meeting_id, data.lang)
        return {"summary": output}
    else:
        raise HTTPException(status_code=401, detail=("access denied"))
