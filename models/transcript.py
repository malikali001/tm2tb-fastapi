from database import Base
from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from .summary import Summary


class Transcript(Base):
    __tablename__ = "transcript"
    id = Column(Integer, primary_key=True)
    meeting_id = Column(String)
    language_code = Column(String)
    interpreter_audio = Column(Boolean)
    text = Column(String)
    meta_data = Column(String)
    start_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    end_timestamp = Column(DateTime(timezone=True), onupdate=func.now())

    @classmethod
    def create(cls, db: Session, data):
        db_transcript = cls(
            meeting_id=data.meeting_id,
            language_code=data.lang_code,
            interpreter_audio=data.interpreter_audio,
            text=data.text,
            meta_data=data.text,
        )
        db.add(db_transcript)
        db.commit()
        db.refresh(db_transcript)
        return db_transcript

    @classmethod
    def fetch_summary_transcript(cls, db: Session, summary: Summary):
        transcript_data = db.query(cls).filter(
            cls.meeting_id == summary.meeting_id,
            cls.language_code == summary.language_code,
            cls.start_timestamp > summary.timestamp,
        )
        return transcript_data

    @classmethod
    def fetch(cls, db: Session, meeting_id: str, lang: str):
        transcript_data = db.query(cls).filter(
            cls.meeting_id == meeting_id, cls.language_code == lang
        )
        return transcript_data
