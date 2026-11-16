# Note technique — étape 1

## Périmètre et exigences

MaintiDoc est un assistant documentaire de maintenance. Cette livraison
couvre uniquement le RAG multimodal agentique demandé dans la partie I.

| Exigence | Réalisation |
| --- | --- |
| PDF texte, images et tableaux | Article du cours et manuel fictif fourni dans `examples/` |
| LlamaIndex et PGVector | PyMuPDFReader, Document/MediaResource, TextNode/ImageNode, PGVectorStore |
| Modèle d'embeddings ouvert | BGE-M3 via TEI ; OpenCLIP pour les visuels |
| Descriptions indexées | Qwen3-VL, contrôle automatique, découpage puis BGE-M3 |
| RAG avec LLM local | Qwen3 Instruct via Ollama, contexte documentaire et citations |
| Streamlit | Chargement, indexation, questions, réponses et sources |
| Agent LangChain | Choix JSON du premier outil, boucle bornée, contexte et génération finale |

## Choix des modèles et outils

- **BGE-M3, 1024 dimensions** : recherche sémantique multilingue, notamment
  questions françaises sur un document anglais. TEI expose une API locale
  et conserve le modèle dans un volume Docker.
- **OpenCLIP ViT-B-32 / laion2b_s34b_b79k, 512 dimensions** : comparaison
  texte-image dans un espace partagé. Les images sont compositées sur fond
  blanc et complétées en carré avant le prétraitement CLIP, pour conserver
  les bords des schémas. Les requêtes CLIP sont formulées en anglais.
- **Qwen3 4B Instruct, Q4_K_M** : sélection des outils et génération textuelle
  sur une machine disposant de ressources limitées. Le tag Instruct évite
  le comportement de raisonnement rencontré avec un précédent tag.
- **Qwen3-VL 2B Instruct, Q4_K_M** : descriptions des visuels et observations
  structurées pour le contrôle automatique.
- **Streamlit** : interface Python adaptée au périmètre pédagogique, sans
  serveur REST supplémentaire. LangChain utilise ChatOllama ; LlamaIndex
  gère les documents et l'accès vectoriel.

Les API de TEI et d'Ollama sont locales. Les premières installations exigent
un téléchargement des dépendances et des poids. PostgreSQL et TEI sont dans
Docker ; Python utilise `.venv` ; Ollama fonctionne nativement sur Windows.

## Données, recherche et provenance

Les passages sont découpés avec `SentenceSplitter` : taille 512, chevauchement
64, selon son tokenizer. Ce nombre n'est pas une dimension vectorielle. Le
tokenizer BGE diffère ; TEI refuse les entrées trop longues au lieu de les
tronquer silencieusement. Le service CPU actuel est configuré à 1024 tokens.

La table `public.data_maintidoc_chunks` contient le texte et les descriptions
retenues, avec leurs vecteurs 1024 et métadonnées. La table
`public.data_maintidoc_visuals_clip` contient les références aux images et
leurs vecteurs 512. Les espaces restent séparés. La recherche utilise la
distance cosinus via PGVectorStore ; aucun index HNSW n'est configuré pour
ces petits corpus.

Les références comportent le document, la page, le nœud et, pour les visuels,
le chemin local et un identifiant visuel. L'interface filtre les requêtes
sur le PDF actif et réinitialise la conversation quand ce PDF change.

Une capture de tableau est indexée comme visuel. Ses cellules extraites sont
conservées en JSON et son contenu textuel est accessible dans les passages
de la page originale. Cela ne constitue pas une base relationnelle fiable
pour calculer automatiquement sur des tableaux complexes.

## Contrôle automatique des descriptions

Qwen3-VL observe l'image sans recevoir la description candidate. Qwen3 texte
compare ensuite ces observations à la description. Les statuts sont
`auto_accepted`, `auto_rejected`, `uncertain` ou `error`. Seul le premier est
indexé dans l'espace texte. Les autres restent dans le rapport local.

Ce contrôle n'exige aucune approbation humaine image par image. Il n'est pas
une certification : l'observateur et le juge peuvent se tromper. Les anciennes
annotations de `data/reviews/` sont des notes conservées et ne bloquent pas
le pipeline automatique.

## Validation disponible

Les tests hors ligne vérifient la boucle LangChain avec des modèles simulés,
les identifiants de citation, le périmètre documentaire, les anciens nœuds
sans `source_type`, la reprise du pipeline, les collisions de noms et les
interactions principales du client avec Streamlit AppTest.

Le pipeline complet a également été exécuté avec les vrais services sur
`examples/maintenance_demo.pdf` : 2 pages, 2 passages, 1 figure, 1 tableau et
2 descriptions retenues. Une seconde exécution a réutilisé les artefacts et
revérifié les imports en base.

Le script `scripts/evaluate.py` produit un rapport local contenant les réponses,
leurs sources et les critères vérifiés. Les critères portent sur les outils,
les pages, quelques mots attendus, les refus et l'isolation des documents.
Ils ne constituent pas une évaluation exhaustive de la fidélité des réponses.

Lors de la validation locale du 16 novembre 2026 : **10 tests hors ligne et
7 scénarios avec les vrais services ont réussi**. Streamlit a répondu HTTP 200
sur son endpoint de santé et le rendu AppTest a affiché les deux documents
indexés sans exception.
Un échange réel via AppTest a également appelé l'agent sur le manuel MX-10,
affiché la source de la page 2 et son image ; l'activation de la page originale
a ajouté un second élément image sans exception. Le téléversement lui-même
n'a pas été simulé dans un navigateur : son traitement a été vérifié via
l'appel direct au même pipeline d'indexation.

Exemples observés :

| Question | Résultat observé | Référence |
| --- | --- | --- |
| Sens du voyant STATUS orange du MX-10 | Inspection du filtre au prochain arrêt autorisé | Manuel fictif, page 1 |
| Libellés du panneau MX-10 | POWER, STATUS, STOP | Description automatique, page 2 |
| Afficher le panneau MX-10 | Visuel candidat retrouvé avec CLIP | Manuel fictif, page 2 |
| Capitale de l'Italie | Refus, sans source présentée | Information absente du manuel |
| Nombre de têtes du Transformer avec le manuel MX-10 actif | Refus malgré la présence de l'article dans la base | Isolation documentaire |
| Multi-head attention | Projections, calculs parallèles, concaténation | Article du cours, pages 4 et 5 |
| Libellés du schéma du Transformer | Inputs, Outputs, embeddings, Softmax, Linear | Description automatique, page 3 |

## Limites

- Les citations connues ne prouvent pas à elles seules la justesse d'une
  affirmation. L'appel de génération séparé et les tests hors contexte
  réduisent ce risque sans le supprimer.
- CLIP retrouve des candidats ; il ne lit pas une image. Une description
  automatique peut mal interpréter un schéma, même après le contrôle.
- PyMuPDF peut fusionner des cellules ou perdre des exposants. Les valeurs
  doivent être confrontées à la page originale avant toute décision.
- Les PDF sans texte extractible nécessitent de l'OCR, absent de cette
  version. Certains dessins vectoriels intégrés à une page ne sont pas
  récupérés comme images indépendantes.
- Le traitement local peut prendre plusieurs minutes. Les modèles sont
  déchargés après les appels pour limiter la mémoire, au prix de latence.
- La langue et le format ne sont pas toujours parfaitement respectés : un
  essai de localisation de visuel a répondu en anglais à une question en
  français. Certaines formules peuvent nécessiter une mise en forme adaptée.
- La version est destinée à une démonstration locale : pas d'authentification,
  d'isolation multi-utilisateur de production ni de déploiement public.
- Les empreintes permettent de reprendre les mêmes calculs. Une stratégie
  transactionnelle de remplacement de toutes les versions d'un document
  après changement de modèle reste à développer.

Le fine-tuning, MCP, l'historique des interventions, les tickets et les clients
externes appartiennent aux étapes suivantes.
