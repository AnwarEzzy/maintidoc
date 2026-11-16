import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQueryResult

from scripts.agent import SearchRoute, answer_with_agent, build_tools, validate_answer
from scripts.ask_document import NO_CONTEXT_ANSWER


class ToolCallingModel(FakeMessagesListChatModel):
    """Offline scripted model; the real LangChain agent still executes tools."""

    def bind_tools(self, tools, **kwargs):
        return self


def result(node_id, text, **metadata):
    return VectorStoreQueryResult(nodes=[TextNode(
        id_=node_id,
        text=text,
        metadata={"document": "manual.pdf", "page": 3, **metadata},
    )])


class AgentTests(unittest.TestCase):
    def test_unknown_citation_is_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_answer("Une affirmation [S99].", [{"id": "S1"}])

    def test_uncited_answer_is_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_answer("Une affirmation sans source.", [{"id": "S1"}])

    def test_insufficient_context_hides_irrelevant_sources(self):
        response = validate_answer("INSUFFICIENT_CONTEXT", [{"id": "S1"}])
        self.assertEqual(response, {"answer": NO_CONTEXT_ANSWER, "sources": []})

    @patch("scripts.agent.search_chunks")
    def test_tool_keeps_document_scope_and_visual_provenance(self, search):
        search.return_value = result(
            "image-caption", "Inputs, Softmax", source_type="visual_description",
            image_path="data/processed/figure.png",
        )
        sources, trace = {}, []
        tools = build_tools("manual.pdf", sources, trace)
        tools[1].invoke({"query": "labels"})
        search.assert_called_once_with(
            "labels", 1, document="manual.pdf", descriptions_only=True,
        )
        self.assertEqual(list(sources.values())[0]["image_path"], "data/processed/figure.png")
        # A second session gets its own registry, not the first session's sources.
        other_sources = {}
        build_tools("other.pdf", other_sources, [])
        self.assertEqual(other_sources, {})

    @patch("scripts.agent.search_chunks")
    def test_real_agent_combines_two_tools_and_preserves_source_ids(self, search):
        search.side_effect = [
            result("passage", "The encoder contains six layers."),
            result("caption", "Inputs and Softmax", source_type="visual_description"),
        ]
        model = ToolCallingModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "search_descriptions", "args": {"query": "labels"}, "id": "call2",
            }]),
            AIMessage(content="SEARCH_COMPLETE"),
            AIMessage(content="Six couches [S1]. Selon la description : Inputs [S2]."),
        ])
        router = RunnableLambda(lambda _: SearchRoute(tool="search_passages", query="encoder"))
        response = answer_with_agent(
            "Explique et cite les libellés", "manual.pdf", model=model, router=router,
        )
        self.assertEqual([s["id"] for s in response["sources"]], ["S1", "S2"])
        self.assertEqual([t["tool"] for t in response["trace"]], [
            "search_passages", "search_descriptions",
        ])


if __name__ == "__main__":
    unittest.main()
