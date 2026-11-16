# Démonstration de l'étape 1

## Préparer

1. Démarrer les conteneurs PostgreSQL et embeddings dans Docker Desktop.
2. Vérifier qu'Ollama fonctionne avec les deux tags indiqués dans le README.
3. Lancer `uv run streamlit run app.py` et ouvrir l'URL locale.
4. Charger `examples/maintenance_demo.pdf` puis cliquer sur **Indexer ce PDF**.
5. Observer la fin du traitement et sélectionner le document s'il ne l'est pas.

Le manuel MX-10 décrit une machine fictive. Il contient du texte, un tableau
et une image pour démontrer le logiciel sans donner de procédure réelle.

## Montrer les fonctionnalités

| Question | Ce qu'il faut montrer |
| --- | --- |
| Que signifie le voyant STATUS orange du MX-10 ? | Passage original, sens du voyant, page 1 |
| Quels libellés sont visibles sur le panneau de commande MX-10 ? | Outil descriptions, source automatique, image page 2 |
| Montre l'image du panneau de commande MX-10. | Recherche CLIP, image candidate et référence |
| Quelle est la capitale de l'Italie ? | Refus car le document n'apporte pas la réponse |

Ouvrir les sources, afficher une page originale et télécharger le PDF. Les
réponses exactes peuvent varier : contrôler le sens et les références.

Pour le PDF du cours, demander « What is multi-head attention? » et consulter
les pages 4 et 5. Pour le schéma, demander « Quels libellés sont visibles dans
le schéma du Transformer ? » et vérifier la référence réellement sélectionnée.

## Expliquer le fonctionnement à l'oral

« LlamaIndex représente et découpe les documents. BGE-M3 encode les textes,
CLIP encode les visuels et PostgreSQL avec pgvector conserve les vecteurs et
les références. Un agent LangChain choisit une recherche selon la question.
Qwen reçoit ensuite les sources collectées dans un appel dédié à la réponse.
Streamlit affiche la réponse, les citations et les éléments originaux pour
permettre leur vérification. »

Préciser les limites : descriptions et extraction de tableaux imparfaites,
citations à vérifier, absence de contrôle d'équipement et étapes 2–4 non
implémentées.

## Reproduire les contrôles

```powershell
uv run python -X utf8 -m unittest discover -s tests -v
uv run python -X utf8 -m scripts.evaluate
```

Le rapport des essais réels se trouve dans `data/processed/evaluation.json`.
Les données générées ne sont pas envoyées dans les commits Git.
