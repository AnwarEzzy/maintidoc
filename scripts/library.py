"""Local document catalogue and safe access to stored PDF/image references."""

import json
import os
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "processed" / "library"


@contextmanager
def database():
    load_dotenv(ROOT / ".env")
    password = os.getenv("POSTGRES_PASSWORD")
    if not password:
        raise ValueError("Renseigne POSTGRES_PASSWORD dans .env.")
    url = URL.create(
        "postgresql+psycopg2", username=os.getenv("POSTGRES_USER", "maintidoc"),
        password=password, host="127.0.0.1", port=5433,
        database=os.getenv("POSTGRES_DB", "maintidoc"),
    )
    engine = create_engine(url, connect_args={"connect_timeout": 5})
    try:
        with engine.connect() as connection:
            yield connection
    finally:
        engine.dispose()


def local_asset(relative_path: str) -> Path:
    path = (ROOT / relative_path).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("Référence locale absente ou située hors du projet.")
    return path


def catalog_records() -> list[dict]:
    if not CATALOG.exists():
        return []
    return [json.loads(p.read_text(encoding="utf-8")) for p in CATALOG.glob("*.json")]


def list_documents() -> list[dict]:
    manifests = {r["document"]: r for r in catalog_records()}
    with database() as connection:
        if not inspect(connection).has_table("data_maintidoc_chunks"):
            return []
        rows = connection.execute(text("""
            SELECT metadata_->>'document' AS document,
                   count(*) FILTER (WHERE metadata_->>'source_type'
                       = 'visual_description') AS descriptions,
                   count(*) FILTER (WHERE COALESCE(metadata_->>'source_type',
                       'pdf_text') = 'pdf_text') AS passages
            FROM public.data_maintidoc_chunks
            GROUP BY metadata_->>'document' ORDER BY 1
        """))
        documents = []
        for row in rows:
            record = dict(row._mapping)
            manifest = manifests.get(record["document"], {})
            if manifest and manifest.get("status") != "ready":
                continue
            record["display_name"] = manifest.get("display_name", record["document"])
            documents.append(record)
        return documents


def source_pdf(document: str) -> Path | None:
    for record in catalog_records():
        if record["document"] == document:
            return local_asset(record["pdf_path"])
    # Compatibility with the course PDF indexed manually before the dashboard.
    if Path(document).name == document:
        candidate = ROOT.parent / document
        if candidate.is_file() and candidate.suffix.lower() == ".pdf":
            return candidate
    return None


def has_visuals(document: str | None) -> bool:
    with database() as connection:
        if not inspect(connection).has_table("data_maintidoc_visuals_clip"):
            return False
        return bool(connection.execute(text("""
            SELECT EXISTS (SELECT 1 FROM public.data_maintidoc_visuals_clip
            WHERE (:document IS NULL OR metadata_->>'document' = :document))
        """), {"document": document}).scalar_one())
