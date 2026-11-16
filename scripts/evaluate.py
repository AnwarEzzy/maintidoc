"""Live smoke evaluation. Requires running local services and indexed demo PDFs.

These checks cover retrieval routing, document isolation and selected answers;
they are not a general benchmark or a proof that all generated claims are true.
"""

import json
import sys
from pathlib import Path

from scripts.agent import answer_with_agent
from scripts.ask_document import NO_CONTEXT_ANSWER
from scripts.library import ROOT, list_documents


def main():
    documents = list_documents()
    maintenance = next((d["document"] for d in documents if d["display_name"] == "maintenance_demo.pdf"), None)
    if maintenance is None:
        raise SystemExit("Indexe d'abord examples/maintenance_demo.pdf.")
    cases = [
        (maintenance, "Que signifie le voyant STATUS orange du MX-10 ?", "search_passages", 1, "filtre"),
        (maintenance, "Quels libellés sont visibles sur le panneau de commande MX-10 ?", "search_descriptions", 2, "STATUS"),
        (maintenance, "Montre l'image du panneau de commande MX-10.", "search_images", 2, None),
        (maintenance, "Quelle est la capitale de l'Italie ?", "search_passages", None, None),
        # Another document exists in the DB, but cannot supply the active PDF's answer.
        (maintenance, "Combien de têtes d'attention utilise le Transformer de l'article Attention Is All You Need ?", "search_passages", None, None),
    ]
    paper = next((d["document"] for d in documents if d["document"] == "attention is all you need.pdf"), None)
    if paper:
        cases.extend([
            (paper, "What is multi-head attention?", "search_passages", 4, "parallel"),
            (paper, "Quels libellés sont visibles dans le schéma du Transformer ?", "search_descriptions", 3, "Inputs"),
        ])
    results = []
    for document, question, tool, page, word in cases:
        print(f"Question : {question}", flush=True)
        try:
            response = answer_with_agent(question, document=document)
            checks = {
                "expected_first_tool": response["trace"][0]["tool"] == tool,
                "document_isolation": all(s["document"] == document for s in response["sources"]),
            }
            if page is None:
                checks["refusal"] = response["answer"] == NO_CONTEXT_ANSWER and not response["sources"]
            else:
                checks["expected_page"] = any(s["page"] == page for s in response["sources"])
                if word:
                    checks["expected_keyword"] = word.casefold() in response["answer"].casefold()
            record = {"document": document, "question": question, **response,
                      "checks": checks, "passed": all(checks.values())}
        except Exception as error:
            record = {"document": document, "question": question, "error": str(error), "passed": False}
        results.append(record)
        print("PASS" if record["passed"] else "FAIL", flush=True)
    passed = sum(r["passed"] for r in results)
    report = {"passed": passed, "total": len(results), "cases": results}
    output = ROOT / "data" / "processed" / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Résultat : {passed}/{len(results)}. Rapport : {output}")
    if passed != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
