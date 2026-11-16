# Comprendre la partie LangChain de MaintiDoc

## 1. Ce qui change par rapport à notre premier RAG

Le RAG classique exécutait toujours : question → recherche BGE → passages →
Qwen → réponse. Le paramètre `--visual-descriptions` permettait de choisir
manuellement la recherche dans les descriptions.

L'agent choisit maintenant l'outil selon la demande. « Explique l'attention »
appelle la recherche dans le texte. « Quels libellés sont visibles ? » appelle
les descriptions. « Montre le schéma » appelle CLIP. Ce choix vient du modèle,
pas d'une liste de mots-clés codée en Python.

L'agent organise le RAG. Il ne constitue pas une nouvelle base de connaissances.

## 2. Qu'est-ce que LangChain ?

LangChain est une bibliothèque Python qui relie des modèles, des messages et
des outils. Ici, elle fournit l'adaptateur Ollama, les outils typés et la boucle
de l'agent. `create_agent` s'appuie sur LangGraph pour exécuter cette boucle ;
nous n'avons pas construit un second système d'agents séparé.

Qwen reste le modèle qui fait les choix et produit du texte. LangChain exécute
les opérations que notre programme autorise. LlamaIndex garde son rôle de
représentation des documents et de recherche dans PGVector.

## 3. ChatOllama : le même serveur, une interface Python

Tu appelais auparavant `http://127.0.0.1:11434/api/chat` avec PowerShell ou
`urllib`. `ChatOllama` communique avec ce même Ollama et le même modèle Qwen.
Le modèle n'est pas téléchargé une seconde fois par LangChain.

Dans `scripts/agent.py` :

```python
llm = ChatOllama(
    model=MODEL_NAME,
    base_url="http://127.0.0.1:11434",
    temperature=0,
    num_ctx=8192,
    keep_alive=0,
)
```

`num_ctx` désigne la fenêtre de contexte. `keep_alive=0` décharge le modèle
après la requête pour limiter la mémoire occupée. Cela entraîne des rechargements
et augmente la latence. Plusieurs appels dans une question utilisent le même
modèle, sans entraîner ni modifier ses poids.

## 4. Un outil est une fonction Python décrite au modèle

```python
@tool
def search_descriptions(query: str) -> str:
    """Search automatic image descriptions for visible labels and layout."""
    result = search_chunks(
        query, 1, document=document, descriptions_only=True,
    )
    return collect("search_descriptions", query, result)
```

Le décorateur `@tool` expose le nom, la description et le paramètre `query`.
Le modèle choisit un nom et propose les arguments. LangChain appelle ensuite
la fonction Python. C'est cette fonction qui contacte TEI et PGVector.

La fonction capture `document`, fixé par l'application. Le modèle ne peut pas
remplacer ce filtre pour chercher dans un autre PDF.

Les trois outils disponibles sont :

| Outil | Recherche effectuée | Limite |
| --- | --- | --- |
| `search_passages` | Texte original du PDF, BGE-M3 | Une extraction de texte peut perdre la mise en page |
| `search_descriptions` | Descriptions retenues, BGE-M3 | Une description automatique peut être incorrecte |
| `search_images` | Texte anglais vers images, CLIP | La similarité ne prouve pas le contenu de l'image |

## 5. Pourquoi imposer un premier choix JSON ?

Pendant les tests, Qwen répondait parfois directement sans appeler d'outil.
Le prompt seul ne suffisait pas.

`SearchRoute` définit donc un résultat structuré :

```json
{
  "tool": "search_descriptions",
  "query": "libellés du schéma du Transformer"
}
```

Le nom doit appartenir aux trois outils autorisés et la requête doit être
non vide. Le middleware `require_first_search` transforme ce choix validé en
appel d'outil. Le premier passage par l'agent produit ainsi une recherche.

Pour la première question, BGE reçoit le texte original de la question afin
d'éviter une reformulation qui ferait perdre des détails. CLIP reçoit une
requête visuelle en anglais. Pour les questions de suivi, la reformulation
peut utiliser les derniers échanges.

## 6. La boucle de l'agent

Lis dans cet ordre : `SearchRoute`, `build_tools`, puis `answer_with_agent`.

1. L'application fournit la question, le PDF actif et un historique limité.
2. Qwen choisit le premier outil avec un JSON validé.
3. LangChain exécute la fonction et reçoit les résultats.
4. L'agent peut choisir une recherche supplémentaire si elle est utile.
5. Il termine la collecte avec `SEARCH_COMPLETE`.

Le registre `sources` associe chaque nœud à un identifiant stable pendant la
question : `S1`, `S2`, etc. La liste `trace` enregistre les outils réellement
exécutés et leurs requêtes. L'interface affiche cette trace sous « Recherches
effectuées ». Il ne s'agit pas d'un accès au raisonnement interne du modèle.

La boucle est bornée. Les outils servent uniquement à lire les documents.
Ils ne commandent aucun équipement et ne créent aucun ticket.

## 7. Pourquoi séparer la réponse finale ?

Un essai a produit « Rome » avec une citation vers le PDF des Transformers.
Le numéro de source existait, mais il ne justifiait pas l'affirmation.

Le dialogue de recherche n'est donc pas utilisé comme réponse. Un nouvel
appel `llm.invoke` reçoit seulement le prompt RAG, la question et les sources
collectées. Les plans et les suppositions de l'agent restent en dehors de ce
contexte. Cet appel utilise toujours LangChain et Qwen via Ollama.

`validate_answer` vérifie ensuite : réponse non vide, citations présentes et
identifiants connus. `INSUFFICIENT_CONTEXT` devient notre message de refus.
Ces contrôles améliorent le résultat ; ils ne prouvent pas que chaque phrase
est une conséquence exacte du document. Les pages restent consultables pour
vérifier les réponses.

## 8. Expériences pour maîtriser le fonctionnement

Dans Streamlit, sélectionne le manuel fictif puis demande successivement :

1. « Que signifie le voyant STATUS orange du MX-10 ? » Observe l'outil texte.
2. « Quels libellés sont visibles sur le panneau MX-10 ? » Observe les descriptions.
3. « Montre l'image du panneau MX-10. » Observe CLIP et le visuel affiché.
4. « Quelle est la capitale de l'Italie ? » Observe le refus.
5. Demande une information présente uniquement dans un autre PDF. Le filtre
   doit empêcher de l'utiliser pour répondre sur le manuel actif.

Pour chaque essai, ouvre « Sources vérifiables » et « Recherches effectuées ».
Tu peux ensuite comparer la trace aux fonctions dans `scripts/agent.py`.

Références : [Agents LangChain](https://docs.langchain.com/oss/python/langchain/agents),
[ChatOllama](https://docs.langchain.com/oss/python/integrations/chat/ollama).
