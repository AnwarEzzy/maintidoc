import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


project_root = Path(__file__).resolve().parents[1]
load_dotenv(project_root / ".env")

password = os.getenv("POSTGRES_PASSWORD")

if not password:
    raise SystemExit("POSTGRES_PASSWORD is missing from the environment.")

database_url = URL.create(
    drivername="postgresql+psycopg2",
    username=os.getenv("POSTGRES_USER", "maintidoc"),
    password=password,
    host="127.0.0.1",
    port=5433,
    database=os.getenv("POSTGRES_DB", "maintidoc"),
)

engine = create_engine(
    database_url,
    connect_args={"connect_timeout": 10},
)

try:
    with engine.connect() as connection:
        database_name = connection.execute(
            text("SELECT current_database()")
        ).scalar_one()

        vector_version = connection.execute(
            text(
                "SELECT extversion FROM pg_extension "
                "WHERE extname = 'vector'"
            )
        ).scalar_one_or_none()

        if vector_version is None:
            raise SystemExit("The pgvector extension is not enabled.")

        print(f"Database connection: OK ({database_name})")
        print(f"pgvector extension: OK ({vector_version})")
finally:
    engine.dispose()