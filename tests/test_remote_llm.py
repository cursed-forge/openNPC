"""Tests of the remote engine (CursedForge fork). Standard library only, so
they run with ``python -m unittest`` and with pytest."""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from opennpc.llm import LLMEngine, create_engine
from opennpc.remote_llm import RemoteLLMEngine, RemoteLLMError


class _Stub(BaseHTTPRequestHandler):
    models = ["pinned-model", "other-model"]
    answer_model = "pinned-model"
    seen: list[dict] = []
    status_code = 200

    def log_message(self, *args):  # silence the test output
        pass

    def _send(self, status, body):
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.headers.get("Authorization") != "Bearer secret":
            self._send(401, {"detail": "no"})
            return
        self._send(200, {"data": [{"id": m} for m in type(self).models]})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        body["_ua"] = self.headers.get("User-Agent")
        type(self).seen.append(body)
        if type(self).status_code != 200:
            self._send(type(self).status_code, {"detail": "fail"})
            return
        self._send(200, {
            "model": type(self).answer_model,
            "choices": [{"message": {"role": "assistant", "content": '  {"ok": true}  '}}],
            "usage": {"completion_tokens": 7},
        })


class RemoteEngineTests(unittest.TestCase):
    def setUp(self):
        _Stub.models = ["pinned-model", "other-model"]
        _Stub.answer_model = "pinned-model"
        _Stub.seen = []
        _Stub.status_code = 200
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def engine(self, **kw):
        options = dict(base_url=self.url, model="pinned-model", api_key="secret",
                       chat_path="/chat", models_path="/models", timeout_s=5,
                       extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                       user_agent="Test-UA/1.0")
        options.update(kw)
        return RemoteLLMEngine(**options)

    def test_generate_sends_the_pinned_model_schema_and_extra_body(self):
        engine = self.engine()
        schema = {"type": "json_schema", "json_schema": {"name": "x", "strict": True, "schema": {"type": "object"}}}
        result = engine.generate("hello", system_prompt="rules", response_format=schema, temperature=0.2)
        self.assertEqual(result.text, '{"ok": true}')
        self.assertEqual(result.tokens_generated, 7)
        sent = _Stub.seen[-1]
        self.assertEqual(sent["model"], "pinned-model")
        self.assertEqual(sent["messages"][0], {"role": "system", "content": "rules"})
        self.assertEqual(sent["response_format"], schema)
        self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(sent["temperature"], 0.2)
        self.assertFalse(sent["stream"])
        self.assertEqual(sent["_ua"], "Test-UA/1.0")

    def test_load_refuses_a_server_without_the_pinned_model(self):
        _Stub.models = ["other-model"]
        engine = self.engine()
        with self.assertRaises(RemoteLLMError) as caught:
            engine.load()
        self.assertEqual(caught.exception.code, "model_missing")
        self.assertFalse(engine.is_loaded)

    def test_a_wrong_key_is_an_http_error(self):
        with self.assertRaises(RemoteLLMError) as caught:
            self.engine(api_key="wrong").load()
        self.assertEqual(caught.exception.code, "http_401")

    def test_an_answer_from_another_model_is_refused(self):
        _Stub.answer_model = "paid-model"
        with self.assertRaises(RemoteLLMError) as caught:
            self.engine().generate("hello")
        self.assertEqual(caught.exception.code, "model_mismatch")

    def test_a_server_error_is_counted(self):
        engine = self.engine()
        engine.load()
        _Stub.status_code = 500
        with self.assertRaises(RemoteLLMError) as caught:
            engine.generate("hello")
        self.assertEqual(caught.exception.code, "http_500")
        self.assertEqual(engine.status()["failures"], 1)

    def test_an_unreachable_server_is_a_typed_error(self):
        engine = RemoteLLMEngine(base_url="http://127.0.0.1:9", model="m", timeout_s=1)
        with self.assertRaises(RemoteLLMError) as caught:
            engine.load()
        self.assertIn(caught.exception.code, ("unreachable", "timeout"))

    def test_the_model_is_required(self):
        with self.assertRaises(ValueError):
            RemoteLLMEngine(base_url=self.url, model="")

    def test_the_factory_builds_both_backends(self):
        self.assertIsInstance(create_engine("remote", base_url=self.url, model="m"), RemoteLLMEngine)
        self.assertIsInstance(create_engine("local"), LLMEngine)
        with self.assertRaises(ValueError):
            create_engine("cloud")

    def test_the_local_engine_never_trusts_remote_code(self):
        import inspect
        import opennpc.llm as llm

        self.assertNotIn("trust_remote_code=True", inspect.getsource(llm))


if __name__ == "__main__":
    unittest.main()
