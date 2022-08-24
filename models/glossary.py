from database import Base
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import Session


class Glossary(Base):
    __tablename__ = "glossary"
    id = Column(Integer, primary_key=True)
    meeting_id = Column(String)
    source_language_code = Column(String)
    target_language_code = Column(String)
    glossary_json = Column(String)

    @classmethod
    def create(cls, db: Session, meeting_id, source_lang, target_lang, glossary_json):
        db_glossary = cls(
            meeting_id=meeting_id,
            source_language_code=source_lang,
            target_language_code=target_lang,
            glossary_json=str(glossary_json),
        )
        db.add(db_glossary)
        db.commit()
        db.refresh(db_glossary)
        return db_glossary
