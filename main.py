from fastapi import Depends, FastAPI
from fastapi_versioning import VersionedFastAPI, version

from api.v1_0 import glossary_view, summary_view, transcript_view
from database import Base, engine
from dependencies import APIKey, get_api_key

Base.metadata.create_all(bind=engine)

app = FastAPI(title="tm2tb")

app.include_router(transcript_view.router)
app.include_router(summary_view.router)
app.include_router(glossary_view.router)


@app.get("/info")
@version(1, 0)
async def info(api_key: APIKey = Depends(get_api_key)):
    return [
        {"minor version": 39, "details": "Overhaul of ranking calculation"},
        {
            "minor version": 40,
            "details": "Deduplication, lemma collapsation. POS matching: PROPN removed, longer "
            "sequences given priority.",
        },
        {
            "minor version": 41,
            "details": "streamline biterm_extractor to reduce latency",
        },
        {"minor version": 42, "details": "add source term frequencies to results"},
        {"minor version": 43, "details": "fix info endpoint"},
        {"minor version": 44, "details": "collapse lemmas by default"},
        {"minor version": 45, "details": "fix return all unmatched source terms"},
        {"minor version": 46, "details": "optionally MT unmatched source terms"},
        {"minor version": 47, "details": "handle longest spans after finding matches"},
        {"minor version": 48, "details": "add source term clustering"},
        {"minor version": 49, "details": "add summarization endpoint"},
    ]


app = VersionedFastAPI(
    app, version_format="{major}_{minor}", prefix_format="/api/v{major}_{minor}"
)
