from typing import List, Optional

from dependencies import APIKey, JwtAuthentication, get_api_key, get_db
from fastapi import APIRouter, Depends, HTTPException, Request
from helpers import extract_glossary, transcript_data
from models import Glossary
from pydantic import BaseModel
from sqlalchemy.orm import Session
from tm2tb import BitermExtractor


class RequestData(BaseModel):
    """
    Request data model.

    It represents the source and target texts and their language identifiers.

    src_texts : List of strings
        List of strings representing the source texts or sentences.
    tgt_texts : List of strings
        List of strings representing the target texts or sentences.
    src_lang : string
        Optional source language identifier
    tgt_lang : string
        Optional target language identifier
    freq_min : integer
        Optional minimum term frequency value. The default is 1
    similarity_min : float
        Minimum similarity value of source and target terms. The default is 0.9
    filter_stopwords: bool
        Optional boolean value. The default is True.
    collapse_lemmas: bool
        Optional value. The default is True. The results will be reduced by looking at their lower-case lemmas.
    return_unmatched_terms: bool
        Optional value. The default is True. If False, source terms without a target match are not returned.
    mt_unmatched_terms: bool
        Optional value. The default is False. If True, the unmatched src terms are machine translated.
    """

    src_texts: List[str]
    tgt_texts: List[str]
    src_lang: Optional[str] = None
    tgt_lang: Optional[str] = None
    freq_min: Optional[int] = 1
    span_range: Optional[tuple] = (1, 7)
    similarity_min: Optional[float] = 0.9
    filter_stopwords: Optional[bool] = True
    include_entities: Optional[bool] = False
    collapse_lemmas: Optional[bool] = True
    return_unmatched_terms: Optional[bool] = True
    mt_unmatched_terms: Optional[bool] = False


class RequestDataForGlossary(BaseModel):
    """
    Request data model.

    It represents the source and target lang

    source_lang : strings
        strings representing the source lang code
    target_lang : strings
        strings representing the target lang code
    """

    source_lang: str
    target_lang: str


class ResponseData(BaseModel):
    """
    Response data model.

    It represents the extracted biterms and their metadata.

    src_terms :
        List of strings representing the source terms.
    src_labels :
        List of strings representing the source terms entity labels (if any).
    src_frequencies :
        List of integers representing the source terms frequencies.
    src_clusters :
        List of integers representing the clusters of source terms.
    tgt_terms :
        List of strings representing the target terms.
    similarities :
        List of floats representing the biterms similarities.
    frequencies :
        List of integers representing the biterms frequencies.
    ranks :
        List of floats representing the biterms ranks.
    origins :
        List of strings representing the origin of the matches ("similarity" or "MT").

    """

    src_terms: List[str]
    src_labels: List[str]
    src_frequencies: List[int]
    src_clusters: List[int]
    tgt_terms: List[str]
    similarities: List[float]
    frequencies: List[int]
    ranks: List[float]
    origins: List[str]


router = APIRouter()


@router.post("/biterms", response_model=ResponseData)
async def extract_biterms(data: RequestData, api_key: APIKey = Depends(get_api_key)):
    """
    Pass bitext data to the biterm extractor to extract biterms.

    Parameters
    ----------
    data : Object of type RequestData.
        It represents bilingual sentences.

    Returns
    -------
    biterms : dict
        A dictionary representing the extracted terms and metadata.
    """
    try:
        # Make a list of tuples (src_text, tgt_text) from the request data dict
        bitext_data = list(zip(data.src_texts, data.tgt_texts))

        # Instantiate extractor and pass the list of tuples
        extractor = BitermExtractor(
            bitext_data, src_lang=data.src_lang, tgt_lang=data.tgt_lang
        )
        # Extract terms
        biterms = extractor.extract_terms(
            freq_min=data.freq_min,
            span_range=data.span_range,
            filter_stopwords=data.filter_stopwords,
            similarity_min=data.similarity_min,
            include_entities=data.include_entities,
            collapse_lemmas=data.collapse_lemmas,
            return_unmatched_terms=data.return_unmatched_terms,
            mt_unmatched_terms=data.mt_unmatched_terms,
        )

        biterms_dict = biterms.to_dict(orient="list")
    except ValueError as e:
        if str(e) == "No terms found." or str(e) == "No biterms found.":
            biterms_dict = {
                "src_terms": [],
                "src_labels": [],
                "src_frequencies": [],
                "src_clusters": [],
                "tgt_terms": [],
                "frequencies": [],
                "similarities": [],
                "ranks": [],
                "origins": [],
            }
        else:
            raise HTTPException(status_code=422, detail=str(e))

    return biterms_dict


@router.post("/meetings/{meeting_id}/glossary", response_model=ResponseData)
async def glossary(
    data: RequestDataForGlossary,
    meeting_id: str,
    request: Request,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
):
    """
    Fetch source transcript and target transcript and Pass bitext data to the biterm extractor to extract biterms and Store glossray in database.

    Parameters
    ----------
    data : Object of type RequestDataForGlossary.
        It represents source and target langs sentences.

    Returns
    -------
    biterms : dict
        A dictionary representing the extracted terms and metadata.
    """
    access_token = request.headers.get("access_token")
    if JwtAuthentication.secure(access_token, meeting_id):
        source_transcript = transcript_data(db, meeting_id, data.source_lang)
        target_transcript = transcript_data(db, meeting_id, data.target_lang)
        if source_transcript and target_transcript:
            biterms_dict = extract_glossary(
                [source_transcript],
                [target_transcript],
                data.source_lang,
                data.target_lang,
            )
            Glossary.create(
                db, meeting_id, data.source_lang, data.target_lang, biterms_dict
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=("source or target transcript not found for this meeting"),
            )
        return biterms_dict
    else:
        raise HTTPException(status_code=401, detail=("access denied"))
