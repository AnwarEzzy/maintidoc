# MaintiDoc — architecture de l'étape 1

## Composants exécutés

```mermaid
flowchart LR
    U[Technicien] --> S[Client Streamlit - Python]
    S --> I[Indexation - LlamaIndex et PyMuPDF]
    S --> A[Agent LangChain - ChatOllama]
    I --> T[TEI et BGE-M3 - Docker - port 8080]
    I --> C[OpenCLIP et PyTorch - CPU Python]
    I --> O[Ollama natif - Qwen et Qwen-VL - port 11434]
    I --> P[PostgreSQL et PGVector - Docker - port 5433]
    A --> O
    A --> R[Outils de recherche]
    R --> T
    R --> C
    R --> P
```

Streamlit appelle les fonctions Python directement. Les API HTTP appartiennent
à TEI et à Ollama ; nous n'avons pas ajouté de serveur FastAPI.

## Flux d'indexation

```mermaid
flowchart TD
    PDF[PDF chargé] --> SAVE[PDF conservé avec un nom lié à son contenu]
    SAVE --> TEXT[Texte par page - PyMuPDFReader de LlamaIndex]
    TEXT --> CHUNKS[SentenceSplitter - 512 tokens - chevauchement 64]
    CHUNKS --> BGE[BGE-M3 - vecteurs texte 1024]
    BGE --> PG1[PGVector - data_maintidoc_chunks]
    SAVE --> VIS[PyMuPDF - images et tableaux candidats]
    VIS --> DOC[LlamaIndex Document et MediaResource]
    DOC --> CLIP[OpenCLIP - vecteurs image 512]
    CLIP --> PG2[PGVector - data_maintidoc_visuals_clip]
    DOC --> VL[Qwen3-VL - descriptions]
    VL --> CHECK[Observation de l'image puis comparaison automatique]
    CHECK -->|auto_accepted| CAP[Descriptions et références]
    CAP --> CHUNKS
    CHECK -->|autres statuts| REPORT[Rapport conservé - exclusion de l'index texte]
```

Les tableaux sont conservés sous forme de cellules JSON et de captures PNG.
Leur texte est également présent dans les passages des pages du PDF. Leurs
captures sont indexées par CLIP et les descriptions retenues par BGE. La
reconstruction exacte des cellules complexes n'est pas garantie.

`scripts/ingest_pdf.py` enchaîne les scripts déjà écrits. Un manifeste conserve
les empreintes des entrées, du code et des sorties. Une reprise réutilise les
étapes inchangées ; les étapes de stockage sont toujours revérifiées pour
permettre de repeupler une base recréée. Les identifiants des nœuds évitent les
doublons lorsque les mêmes enregistrements sont réimportés.

## Flux de question-réponse

```mermaid
flowchart TD
    Q[Question et PDF actif] --> ROUTE[Qwen via ChatOllama - choix JSON validé]
    ROUTE --> AGENT[Agent LangChain]
    AGENT --> T1[search_passages - texte original]
    AGENT --> T2[search_descriptions - descriptions retenues]
    AGENT --> T3[search_images - candidats CLIP]
    T1 --> CTX[Sources avec identifiants S1, S2...]
    T2 --> CTX
    T3 --> CTX
    CTX --> NEXT[Agent - autre recherche utile ou fin]
    NEXT -->|autre recherche| AGENT
    NEXT -->|fin| GEN[Appel dédié à Qwen - question et sources uniquement]
    GEN --> VALID[Contrôle des citations et de la fin de génération]
    VALID --> UI[Réponse ou refus - extraits et visuels consultables]
```

Le filtre du PDF actif est imposé par le code dans les outils. Le modèle ne
choisit pas le périmètre documentaire. Les scores BGE et CLIP ne sont pas
mélangés : ils appartiennent à deux espaces vectoriels distincts.

L'agent conserve au maximum six sources et effectue au maximum quatre
recherches réussies. La boucle possède aussi une limite de récursion. Les
appels d'outils d'une question sont sérialisés. Les quatre derniers messages
de conversation, limités en taille, servent à résoudre les questions de suivi.
Ils ne sont pas transmis comme preuves à la génération finale.

## Fichiers à lire

| Fonction | Fichier |
| --- | --- |
| Interface | `app.py` |
| Enchaînement de l'indexation | `scripts/ingest_pdf.py` |
| Agent, outils et génération finale | `scripts/agent.py` |
| Recherche BGE / CLIP | `scripts/search_chunks.py` / `scripts/search_visuals.py` |
| Catalogue et références locales | `scripts/library.py` |
| Contrôle automatique des descriptions | `scripts/auto_check_descriptions.py` |
