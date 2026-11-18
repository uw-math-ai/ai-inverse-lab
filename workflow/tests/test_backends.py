"""Live SDK response-shape regressions without paid API calls."""
import json
import unittest
from unittest.mock import Mock

from langchain_core.messages import AIMessage
from workflow.backends import DemoBackend, OpenAIBackend
from workflow.benchmarks import load_benchmark
from workflow.schemas import Proposal


class BackendSerializationTests(unittest.TestCase):
    def test_nested_parsed_model_in_raw_message_is_json_serializable(self):
        messages = [{"role": "user", "content": json.dumps({"benchmark": load_benchmark()})}]
        proposal = Proposal.model_validate(DemoBackend().invoke("CA", Proposal, messages)["parsed"])
        raw = AIMessage(content="structured answer", additional_kwargs={"parsed": proposal},
                        usage_metadata={"input_tokens": 7, "output_tokens": 5, "total_tokens": 12})
        backend = OpenAIBackend.__new__(OpenAIBackend)
        backend.model = Mock()
        backend.model.with_structured_output.return_value.invoke.return_value = {
            "raw": raw, "parsed": proposal, "parsing_error": None}
        response = backend.invoke("CA", Proposal, messages)
        saved = json.loads(json.dumps(response))
        self.assertEqual(saved["raw"]["additional_kwargs"]["parsed"], proposal.model_dump())
        self.assertEqual(saved["usage"]["total_tokens"], 12)


if __name__ == "__main__":
    unittest.main()
