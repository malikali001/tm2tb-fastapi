import os

from fastapi import HTTPException
from sqlalchemy.orm import Session

from models import Summary, Transcript
from tm2tb import BitermExtractor, Summarizer

SUMMARY_THRESHOLD = os.environ.get("SUMMARY_THRESHOLD")


def convert_to_dict(source_terms, target_terms):
    return [
        {"src_term": source_terms, "tgt_term": target_terms}
        for (source_terms, target_terms) in zip(source_terms, target_terms)
    ]


def transcript_data(db: Session, meeting_id, lang):
    transcript = ""
    transcript_data = Transcript.fetch(db, meeting_id, lang)
    if transcript_data.count() > 0:
        return transcript.join(map(lambda x: x.text, transcript_data))
    else:
        return False


def extract_glossary(source_transcript, target_transcript, src_lang, tgt_lang):
    try:
        bitext_data = list(zip(source_transcript, target_transcript))

        extractor = BitermExtractor(bitext_data, src_lang=src_lang, tgt_lang=tgt_lang)

        biterms = extractor.extract_terms(
            freq_min=1,
            span_range=(1, 7),
            filter_stopwords=True,
            similarity_min=0.9,
            include_entities=True,
            collapse_lemmas=True,
            return_unmatched_terms=True,
            mt_unmatched_terms=True,
        )

        return biterms.to_dict(orient="list")
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


def check_timestamp(db: Session, summary):
    """
    Check latest transcript is greater than threshold or not.
    Args:
        db (Session): database session
        summary (query): updated summary query object

    Returns:
        boolean: true or false
    """
    transcripts = Transcript.fetch_summary_transcript(db, summary)
    if transcripts.count() > int(SUMMARY_THRESHOLD):
        return False
    return True


def generate_and_save_summary(db: Session, meeting_id, lang, summary_id=0):
    """
    Fetch transcript from database and check summary_id.
    If summary_id greater than 0,  update summary information in the database accordingly.
    if summary_id not greater than 0, create  summary information in the database accordingly

    Args:
        db (Session): database session
        meeting_id (string): given meeting id
        lang (string): given language code
        summary_id (int, optional): latest database summary id. Defaults to 0.

    Raises:
        HTTPException: if transcript not found with respect to the given meeting_id and language code.

    Returns:
        output : list of summary sentences
    """
    transcript = transcript_data(db, meeting_id, lang)
    if transcript:
        summarizer = Summarizer(
            texts=[transcript],
            lang=lang,
            summary_sentences_n=15,
        )
        output = summarizer.extract_summary()
        summary = " | ".join([str(element) for element in output])
        if summary_id > 0:
            Summary.update(db, summary_id, summary)
        else:
            Summary.create(db, meeting_id, lang, summary)
    else:
        raise HTTPException(
            status_code=404, detail=("Unfortunately transcript not found")
        )
    return output
