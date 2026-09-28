"""Additional boundary and isolation checks; no checkpoint downloads."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya import retrieval_cli
from laya.retrieval import Document, LocalIndex


class BoundaryTests(unittest.TestCase):
    def test_invalid_corpus_limits_fail_before_open(self):
        for name in ("max_chars", "max_documents"):
            for value in (-1, 0, True, 1.5):
                with self.subTest(name=name, value=value), patch("builtins.open") as opened:
                    with self.assertRaises(ValueError):
                        retrieval_cli.load_corpus("unused", **{name: value})
                    opened.assert_not_called()

    def test_exact_corpus_character_limit(self):
        record = json.dumps({"id": "d", "text": "refund"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            path.write_text(record, encoding="utf-8")
            self.assertEqual(len(retrieval_cli.load_corpus(str(path), max_chars=len(record))), 1)
            with self.assertRaises(ValueError):
                retrieval_cli.load_corpus(str(path), max_chars=len(record) - 1)

    def test_lexical_import_does_not_load_inference_dependencies(self):
        script = '''
import socket
import sys
def forbidden(*args, **kwargs):
    raise AssertionError("unexpected network access")
socket.create_connection = forbidden
socket.socket.connect = forbidden
from laya.retrieval import Document, LocalIndex
assert LocalIndex([Document("p", "refund policy")]).search("refund")
assert not any(name in sys.modules for name in ("torch", "transformers", "sentence_transformers"))
'''
        result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True,
                                cwd=Path(__file__).resolve().parents[1], timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_every_budget_keeps_exact_whole_evidence(self):
        docs = [Document("a", "refund duplicate invoice", 'policy"\\a'),
                Document("b", "refund café 中文🙂", "policy-b")]
        index = LocalIndex(docs)
        for budget in range(2, 500):
            result = index.prepare({}, query="refund", max_chars=budget)
            encoded = json.dumps(result.state["retrieved_context"], ensure_ascii=False, separators=(",", ":"))
            self.assertLessEqual(len(encoded), budget)
            self.assertEqual(len(encoded), result.context_chars)
            self.assertEqual(len(result.hits) + result.omitted, 2)
            for hit in result.hits:
                document = next(doc for doc in docs if doc.id == hit.document_id)
                self.assertEqual(hit.text, document.text[hit.start:hit.end])

    def test_cli_budget_abstention_does_not_load_router(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            path.write_text('{"id":"p","text":"refund"}', encoding="utf-8")
            output = io.StringIO()
            with patch.object(retrieval_cli, "_load_questions", return_value=({"q": {}}, "request")), \
                 patch.object(retrieval_cli, "_make_router") as make_router, contextlib.redirect_stdout(output):
                code = retrieval_cli.main(["refund", "--corpus", str(path), "--questions", "q.json",
                                           "--predict", "--max-context-chars", "2"])
            make_router.assert_not_called()
            self.assertEqual(code, 3)
            self.assertEqual(json.loads(output.getvalue())["retrieval"]["status"], "budget_exhausted")


if __name__ == "__main__":
    unittest.main()
