"""One entry point for the existing educational ingestion scripts.

Each PDF gets a content-derived name to prevent collisions between uploads.
File stages are checkpointed; database writes always run (idempotently), so a
previously indexed PDF can also repopulate a recreated PostgreSQL volume.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pymupdf

from scripts.library import CATALOG, ROOT

PIPELINE_VERSION = 1


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def execute(module: str, arguments: list[str]):
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    try:
        process = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", module, *arguments],
            cwd=ROOT, env=environment, capture_output=True,
            encoding="utf-8", errors="replace", timeout=1800,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{module} dépasse 30 minutes. Relance pour reprendre.") from error
    if process.returncode:
        details = (process.stdout + "\n" + process.stderr)[-5000:]
        raise RuntimeError(f"Échec de {module} :\n{details}")


def ingest_pdf(content: bytes, filename: str, on_progress=None) -> dict:
    """Extract, describe, embed and store one PDF. No manual caption approval."""
    if not content or len(content) > 50 * 1024 * 1024:
        raise ValueError("Choisis un PDF non vide de moins de 50 Mo.")
    with pymupdf.open(stream=content, filetype="pdf") as pdf:
        if pdf.needs_pass or not len(pdf):
            raise ValueError("Le PDF doit être lisible et non protégé par mot de passe.")
        page_count = len(pdf)
        if not any(page.get_text().strip() for page in pdf):
            raise ValueError("Ce PDF ne contient pas de texte extractible. L'OCR n'est pas inclus.")

    pdf_hash = hashlib.sha256(content).hexdigest()
    safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(filename).stem).strip("_")[:40] or "document"
    stem = f"{safe_stem}_{pdf_hash[:16]}"
    document = f"{stem}.pdf"
    pdf_path = ROOT / "data" / "uploads" / document
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if not pdf_path.exists() or digest(pdf_path) != pdf_hash:
        pdf_path.write_bytes(content)
    processed = ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    manifest_path = CATALOG / f"{stem}.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if previous.get("pipeline_version") != PIPELINE_VERSION:
        previous = {}
    manifest = {
        **previous, "document": document, "display_name": Path(filename).name,
        "pdf_sha256": pdf_hash, "pdf_path": pdf_path.relative_to(ROOT).as_posix(),
        "pipeline_version": PIPELINE_VERSION, "pages": page_count,
        "status": "indexing", "warnings": [],
    }
    checkpoints = manifest.setdefault("checkpoints", {})
    save_json(manifest_path, manifest)

    def artifact(suffix):
        return processed / f"{stem}{suffix}"

    def notify(message):
        if on_progress:
            on_progress(message)

    def step(name, arguments, inputs, outputs, always=False, dependencies=()):
        module = f"scripts.{name}"
        code_paths = [ROOT / "scripts" / f"{name}.py", *dependencies]
        fingerprint = hashlib.sha256(json.dumps([
            [str(p), digest(p)] for p in [*inputs, *code_paths]
        ], sort_keys=True).encode()).hexdigest()
        checkpoint = checkpoints.get(name, {})
        def output_files():
            files = list(outputs)
            if name in {"extract_images", "extract_tables"} and outputs[0].is_file():
                records = json.loads(outputs[0].read_text(encoding="utf-8"))
                files.extend(ROOT / record["image_path"] for record in records)
            return files

        cached = (
            not always and checkpoint.get("fingerprint") == fingerprint
            and all(p.is_file() and checkpoint.get("outputs", {}).get(p.name) == digest(p)
                    for p in output_files())
        )
        notify(f"{'Réutilisation' if cached else 'Exécution'} : {name}")
        if cached:
            return
        execute(module, [str(a) for a in arguments])
        checkpoints[name] = {
            "fingerprint": fingerprint,
            "outputs": {p.name: digest(p) for p in output_files()},
        }
        save_json(manifest_path, manifest)

    try:
        pages, chunks, embeddings = artifact(".json"), artifact(".chunks.json"), artifact(".chunks.embeddings.json")
        step("extract_pdf", [pdf_path], [pdf_path], [pages])
        step("chunk_documents", [pages], [pages], [chunks])
        step("embed_chunks", [chunks], [chunks], [embeddings])
        step("store_embeddings", [embeddings], [embeddings], [], always=True)

        images, tables = artifact(".images.json"), artifact(".tables.json")
        step("extract_images", [pdf_path], [pdf_path], [images])
        step("extract_tables", [pdf_path], [pdf_path], [tables])
        image_records = json.loads(images.read_text(encoding="utf-8"))
        table_records = json.loads(tables.read_text(encoding="utf-8"))
        visual_paths = [ROOT / r["image_path"] for r in [*image_records, *table_records]]
        visuals = artifact(".visual_documents.json")
        accepted_count = 0

        if visual_paths:
            step("prepare_visual_documents", [images, tables], [images, tables, *visual_paths], [visuals])
            clip_vectors = artifact(".visual_documents.clip_embeddings.json")
            step("embed_visuals", [visuals], [visuals, *visual_paths], [clip_vectors],
                 dependencies=[ROOT / "scripts" / "embed_visuals.py"])
            step("store_visual_embeddings", [clip_vectors], [clip_vectors], [], always=True)

            # New generator code gets a new resumable output, keeping old drafts.
            vision_code = ROOT / "scripts" / "describe_image.py"
            descriptions = artifact(f".descriptions_{digest(vision_code)[:8]}.json")
            checked = descriptions.with_name(f"{descriptions.stem}.auto_checked.json")
            step("describe_visuals", [visuals, "--output", descriptions],
                 [visuals, *visual_paths], [descriptions], dependencies=[vision_code])
            step("auto_check_descriptions", [descriptions], [descriptions, *visual_paths], [checked])
            checks = json.loads(checked.read_text(encoding="utf-8"))
            accepted_count = sum(r["description_status"] == "auto_accepted" for r in checks)
            excluded = len(checks) - accepted_count
            if excluded:
                manifest["warnings"].append(f"{excluded} description(s) écartée(s) par le contrôle automatique.")
            if accepted_count:
                indexable = artifact(".visual_descriptions.json")
                caption_chunks = artifact(".visual_descriptions.chunks.json")
                caption_vectors = artifact(".visual_descriptions.chunks.embeddings.json")
                step("prepare_description_documents", [checked, indexable], [checked], [indexable])
                # Same modules used twice: separate checkpoint names by their input path.
                saved_checkpoints = checkpoints
                checkpoints = manifest.setdefault("caption_checkpoints", {})
                step("chunk_documents", [indexable], [indexable], [caption_chunks])
                step("embed_chunks", [caption_chunks], [caption_chunks], [caption_vectors])
                step("store_embeddings", [caption_vectors], [caption_vectors], [], always=True)
                checkpoints = saved_checkpoints
            else:
                manifest["warnings"].append("Aucune description retenue ; texte et recherche CLIP disponibles.")

        manifest.update({
            "status": "ready", "passages": len(json.loads(chunks.read_text(encoding="utf-8"))),
            "figures": len(image_records), "tables": len(table_records),
            "accepted_descriptions": accepted_count,
        })
        manifest.pop("error", None)
        save_json(manifest_path, manifest)
        notify("Indexation terminée.")
        return manifest
    except Exception as error:
        manifest.update(status="error", error=str(error))
        save_json(manifest_path, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description="Run the complete multimodal PDF indexing pipeline.")
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()
    try:
        report = ingest_pdf(args.pdf.read_bytes(), args.pdf.name, lambda s: print(s, flush=True))
    except Exception as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
