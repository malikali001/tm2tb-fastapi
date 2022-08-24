from database import Base
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Session, relationship
from sqlalchemy.sql import func


class Summary(Base):
    __tablename__ = "summary"
    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(String)
    language_code = Column(String)
    text = Column(String)
    timestamp = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    glossary_id = Column(Integer, ForeignKey("glossary.id"))
    glossary = relationship("Glossary")

    @classmethod
    def create(cls, db: Session, meeting_id, lang, text):
        db_summary = cls(meeting_id=meeting_id, language_code=lang, text=text)
        db.add(db_summary)
        db.commit()
        db.refresh(db_summary)
        return db_summary

    @classmethod
    def fetch(cls, db: Session, meeting_id, lang):
        db_summary = (
            db.query(cls)
            .filter(cls.meeting_id == meeting_id, cls.language_code == lang)
            .first()
        )
        return db_summary

    @classmethod
    def update(cls, db: Session, id, summary):
        db_summary = db.query(cls).filter(cls.id == id).update({cls.text: summary})
        db.flush()
        db.commit()
        return db_summary
