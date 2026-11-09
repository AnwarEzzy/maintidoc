# Document Indexing Pipeline

## Purpose

Prepare PDF contents for retrieval while preserving references
to the original document and pages.

## Proposed pipeline

```mermaid
flowchart TD
    A["PDF document"] --> B["Extract text, tables, figures and source metadata"]
    B --> C["Prepare text chunks, table representations and image descriptions"]
    C --> D["Generate embeddings"]
    D --> E["Store contents, vectors and metadata in PostgreSQL with pgvector"]
```

## Processing principles

- Split text into passages that remain linked to their source pages.
- Preserve table headers and relationships between rows and columns.
- Generate image descriptions using a vision-language model.
- Generate text embeddings for passages, tables and image descriptions.
- Generate CLIP embeddings for figures.
- Keep text and CLIP vectors in separate vector spaces.
- Retain original PDFs and extracted figures for source consultation.

## Status

Proposed design. Implementation has not started.