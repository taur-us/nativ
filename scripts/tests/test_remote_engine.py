"""Coverage for forwarding models to an external OpenAI-compatible engine."""
import ast
import asyncio
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / "PythonDistribution/Overlay/nativ_remote_engine.py"
SERVER_SOURCE = SOURCE.with_name("nativ_server.py")
NO_HUB_CACHE = "/nonexistent/huggingface/hub"
ENGINE = {"base_url": "http://127.0.0.1:8091/v1", "model": "engine-model", "api_key": "secret"}


class StubResponse:
    """Stands in for the fastapi response classes: keeps what it was built with."""

    def __init__(self, content=b"{}", status_code=200, headers=None, **kwargs):
        self.content = self.body_iterator = content
        self.status_code = status_code
        self.headers = headers or {}
        self.media_type = kwargs.get("media_type")


class StubHTTPError(Exception):
    pass


class StubTransportError(StubHTTPError):
    pass


class StubInvalidURL(Exception):
    pass


class StubHTTPException(Exception):
    """Stands in for fastapi.HTTPException."""

    def __init__(self, status_code, detail=None):
        super().__init__(detail)
        self.status_code, self.detail = status_code, detail


class Pending:
    """A scripted post that does not answer until released, like an engine still reading a long prompt."""

    def __init__(self):
        self.release, self.cancelled = asyncio.Event(), False

    async def wait(self):
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return SimpleNamespace(raise_for_status=lambda: None)


class StubClient:
    """Plays back a script of upstream responses or errors, recording what was sent."""

    sent = []
    script = []
    posted = []
    post_script = []
    metrics_text = ""

    def __init__(self, *args, **kwargs):
        pass

    def build_request(self, method, url, headers=None, json=None):
        return SimpleNamespace(method=method, url=url, headers=headers, json=json)

    async def get(self, url, headers=None, timeout=None):
        return SimpleNamespace(text=StubClient.metrics_text, raise_for_status=lambda: None)

    async def post(self, url, json=None, headers=None):
        StubClient.posted.append(SimpleNamespace(url=url, json=json, headers=headers))
        step = StubClient.post_script.pop(0)
        if isinstance(step, Pending):
            return await step.wait()
        if isinstance(step, Exception):
            raise step
        return step

    async def send(self, request, stream=False):
        StubClient.sent.append(request)
        step = StubClient.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class UpstreamResponse:
    def __init__(self, body=b'{"ok": true}', status_code=200, content_type="application/json", chunks=(), read_error=None):
        self.status_code = status_code
        self.headers = {"content-type": content_type, "content-encoding": "gzip", "date": "now",
                        "server": "engine", "x-extra": "1"}
        self._body, self._chunks, self._read_error = body, chunks, read_error
        self.closed = 0

    async def aread(self):
        if self._read_error:
            raise self._read_error
        return self._body

    async def aclose(self):
        self.closed += 1

    async def aiter_raw(self):
        for chunk in self._chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


def load_module():
    stubs = {
        "httpx": SimpleNamespace(
            AsyncClient=StubClient, Timeout=lambda *a, **k: None, HTTPError=StubHTTPError,
            TransportError=StubTransportError, InvalidURL=StubInvalidURL, Response=UpstreamResponse,
        ),
        "fastapi": SimpleNamespace(HTTPException=StubHTTPException),
        "huggingface_hub": SimpleNamespace(constants=SimpleNamespace(HF_HUB_CACHE=NO_HUB_CACHE)),
        "fastapi.responses": SimpleNamespace(JSONResponse=StubResponse, Response=StubResponse, StreamingResponse=StubResponse),
    }
    spec = importlib.util.spec_from_file_location("nativ_remote_engine_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class EnginesFileTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def write(self, text):
        path = os.path.join(self.directory.name, "engines.json")
        Path(path).write_text(text)
        return path

    def test_missing_file_means_no_engines(self):
        self.assertEqual(self.module.load_engines(os.path.join(self.directory.name, "none.json")), {})

    def test_unreadable_json_means_no_engines(self):
        self.assertEqual(self.module.load_engines(self.write("{not json")), {})

    def test_entries_without_a_base_url_are_dropped(self):
        path = self.write(json.dumps({"a": {"base_url": "http://x/v1"}, "b": {"model": "m"}, "c": 3}))
        self.assertEqual(list(self.module.load_engines(path)), ["a"])

    def test_a_model_whose_entry_is_unusable_is_still_listed(self):
        path = self.write(json.dumps({"qwen": {"base-url": "http://x/v1"}}))
        self.assertEqual(list(self.module.listed_models(path)), ["qwen"])

    def test_a_file_that_turns_unreadable_keeps_what_it_listed(self):
        path = self.write(json.dumps({"qwen": ENGINE}))
        self.module.load_engines(path)
        for broken in ("{half way through an edit", "", "[]"):
            with self.subTest(broken=broken):
                self.write(broken)
                self.assertEqual(self.module.load_engines(path), {"qwen": ENGINE})
                self.assertEqual(list(self.module.listed_models(path)), ["qwen"])

    def test_a_file_never_read_successfully_leaves_the_list_unknown(self):
        for broken in ("{not json", "[]"):
            with self.subTest(broken=broken):
                self.assertIsNone(self.module.listed_models(self.write(broken)))

    def test_removing_the_file_lists_nothing(self):
        path = self.write(json.dumps({"qwen": ENGINE}))
        self.module.load_engines(path)
        os.remove(path)
        self.assertEqual(self.module.listed_models(path), {})
        self.assertEqual(self.module.load_engines(path), {})


class ListedWeightsTests(unittest.TestCase):
    """A listed model is the same model however a request names its files."""

    LISTED = "local/Qwen"

    def setUp(self):
        self.module = load_module()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        # Like the real layout: the Hugging Face cache folder is a link into the engine's own cache.
        self.weights = self.root / "engine" / "models--local--Qwen"
        self.snapshot = self.weights / "snapshots" / "abc"
        self.snapshot.mkdir(parents=True)
        (self.snapshot / "model.safetensors").write_text("weights")
        self.cache = self.root / "hub"
        self.cache.mkdir()
        self.cached = self.cache / "models--local--Qwen"
        self.cached.symlink_to(self.weights)
        self.module.hub_constants = SimpleNamespace(HF_HUB_CACHE=str(self.cache))
        self.engines = {self.LISTED: ENGINE}

    def assert_listed(self, *sources):
        for source in sources:
            with self.subTest(source=str(source)):
                self.assertEqual(self.module.listed_model(source, self.engines), self.LISTED)

    def test_the_id_in_another_letter_case_is_the_listed_model(self):
        self.assert_listed("LOCAL/qwen", "local/QWEN")

    def test_a_folder_or_file_among_its_weights_is_the_listed_model(self):
        self.assert_listed(str(self.cached / "snapshots" / "abc"), str(self.cached), str(self.snapshot),
                           self.snapshot, str(self.snapshot / "model.safetensors"))

    def test_a_link_into_its_weights_is_the_listed_model(self):
        (self.root / "shortcut").symlink_to(self.snapshot)
        (self.cache / "models--org--Alias").symlink_to(self.weights)
        self.assert_listed(str(self.root / "shortcut"), "org/Alias")

    def test_other_models_folders_and_values_are_not(self):
        (self.cache / "models--local--Other").mkdir()
        for source in ("local/Other", "unknown/model", str(self.cache), str(self.root / "engine"), "", None, 3):
            with self.subTest(source=source):
                self.assertIsNone(self.module.listed_model(source, self.engines))

    def test_a_listed_model_with_no_files_here_still_matches_its_id(self):
        self.assertEqual(self.module.listed_model("ORG/remote", {"org/remote": ENGINE}), "org/remote")
        self.assertIsNone(self.module.listed_model("local/Qwen", {"org/remote": ENGINE}))

    def test_its_snapshot_path_is_forwarded_to_its_engine(self):
        self.assertIs(self.module.engine_for({"model": str(self.snapshot)}, self.engines), ENGINE)

    def test_its_snapshot_path_is_not_preloaded(self):
        for arguments in (["--model", str(self.snapshot)], [f"--model={self.snapshot}"]):
            with self.subTest(arguments=arguments):
                self.assertEqual(self.module.without_remote_preload(arguments, self.engines), [])

    def test_its_snapshot_path_is_not_loaded(self):
        self.module.listed_models = lambda *path: self.engines
        with self.assertRaises(StubHTTPException) as refused:
            self.module.require_loadable(self.snapshot)
        self.assertEqual(refused.exception.status_code, 501)
        self.assertIn(self.LISTED, refused.exception.detail)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.engines = {"qwen": ENGINE}

    def test_a_mapped_model_routes_to_its_engine(self):
        self.assertIs(self.module.engine_for({"model": "qwen"}, self.engines), ENGINE)

    def test_other_models_stay_in_process(self):
        for payload in ({"model": "other"}, {}, {"model": 5}, None, []):
            self.assertIsNone(self.module.engine_for(payload, self.engines))

    def test_paths_map_with_and_without_the_v1_prefix(self):
        self.assertEqual(self.module.upstream_path("/v1/chat/completions"), "/chat/completions")
        self.assertEqual(self.module.upstream_path("/responses"), "/responses")

    def test_compaction_is_not_forwarded(self):
        self.assertIsNone(self.module.upstream_path("/v1/responses/compact"))


class JSONReply:
    """The part of an httpx response that count_input_tokens reads."""

    def __init__(self, data, error=None):
        self._data, self._error = data, error

    def raise_for_status(self):
        if self._error:
            raise self._error

    def json(self):
        return self._data


class CountTokensTests(unittest.TestCase):
    payload = {
        "model": "qwen", "instructions": "Be brief.",
        "input": [
            {"type": "message", "role": "user", "content": "hi", "reasoning_content": "dropped"},
            {"type": "message", "role": "tool", "tool_call_id": "c1", "content": "42"},
        ],
        "tools": [{"type": "function", "function": {"name": "f"}}],
    }

    def setUp(self):
        self.module = load_module()
        StubClient.posted, StubClient.post_script = [], []

    def count(self, payload=None):
        return asyncio.run(self.module.count_input_tokens(payload or self.payload, ENGINE))

    def test_tokens_are_counted_by_the_engines_template_then_tokenizer(self):
        StubClient.post_script = [JSONReply({"prompt": "<rendered>"}), JSONReply({"tokens": [1, 2, 3, 4]})]
        response = self.count()
        self.assertEqual(response.content, {"input_tokens": 4})
        template, tokenize = StubClient.posted
        self.assertEqual(template.url, "http://127.0.0.1:8091/apply-template")
        self.assertEqual(tokenize.url, "http://127.0.0.1:8091/tokenize")
        self.assertEqual(tokenize.json, {"content": "<rendered>", "add_special": False})
        self.assertEqual(template.headers, {"authorization": "Bearer secret"})

    def test_instructions_lead_and_only_conversation_fields_are_sent(self):
        StubClient.post_script = [JSONReply({"prompt": "p"}), JSONReply({"tokens": []})]
        self.count()
        sent = StubClient.posted[0].json
        self.assertEqual(sent["messages"], [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "c1", "content": "42"},
        ])
        self.assertEqual(sent["tools"], self.payload["tools"])

    def test_tools_are_left_out_when_the_request_has_none(self):
        StubClient.post_script = [JSONReply({"prompt": "p"}), JSONReply({"tokens": []})]
        self.count({"model": "qwen", "input": []})
        self.assertNotIn("tools", StubClient.posted[0].json)

    def test_an_engine_failure_is_a_503(self):
        StubClient.post_script = [JSONReply({}, error=StubHTTPError("boom"))]
        self.assertEqual(self.count().status_code, 503)

    def test_an_unexpected_engine_reply_is_a_503(self):
        StubClient.post_script = [JSONReply({"unexpected": 1})]
        self.assertEqual(self.count().status_code, 503)


class PreloadTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.engines = {"qwen": ENGINE}

    def strip(self, *arguments):
        return self.module.without_remote_preload(list(arguments), self.engines)

    def test_a_mapped_model_is_not_preloaded(self):
        self.assertEqual(self.strip("--port", "8080", "--model", "qwen", "--host", "x"), ["--port", "8080", "--host", "x"])
        self.assertEqual(self.strip("--model=qwen", "--port", "8080"), ["--port", "8080"])

    def test_other_models_and_arguments_are_untouched(self):
        arguments = ["--model", "other", "--stt-model", "qwen", "--port", "8080"]
        self.assertEqual(self.strip(*arguments), arguments)

    def test_a_trailing_model_flag_is_kept(self):
        self.assertEqual(self.strip("--port", "8080", "--model"), ["--port", "8080", "--model"])


class ResponsesTranslationTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def convert(self, **body):
        return self.module.responses_body_for_engine(body)

    def test_a_tool_message_becomes_a_function_call_output(self):
        result = self.convert(input=[{"type": "message", "role": "tool", "tool_call_id": "c1", "content": "42"}])
        self.assertEqual(result["input"], [{"type": "function_call_output", "call_id": "c1", "output": "42"}])

    def test_an_assistant_message_with_tool_calls_becomes_function_call_items(self):
        call = {"index": 0, "id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"p": 1}'}}
        result = self.convert(input=[{"type": "message", "role": "assistant", "content": None, "tool_calls": [call]}])
        self.assertEqual(result["input"], [{"type": "function_call", "call_id": "c1", "name": "read_file", "arguments": '{"p": 1}'}])

    def test_assistant_text_alongside_tool_calls_is_kept(self):
        call = {"id": "c1", "function": {"name": "f"}}
        result = self.convert(input=[{"type": "message", "role": "assistant", "content": "Checking", "tool_calls": [call]}])
        self.assertEqual([item["type"] for item in result["input"]], ["message", "function_call"])
        self.assertEqual(result["input"][1]["arguments"], "{}")

    def test_plain_messages_lose_chat_extensions_but_keep_their_text(self):
        item = {"type": "message", "role": "assistant", "content": "hi", "reasoning_content": "thought", "name": "x"}
        self.assertEqual(self.convert(input=[item])["input"], [{"type": "message", "role": "assistant", "content": "hi"}])

    def test_text_parts_collapse_to_a_string_and_image_parts_become_input_images(self):
        parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        self.assertEqual(self.module._message_content(parts), "ab")
        image = [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:x"}}]
        self.assertEqual(self.module._message_content(image)[1], {"type": "input_image", "image_url": "data:x"})

    def test_nested_tool_definitions_are_flattened(self):
        nested = {"type": "function", "function": {"name": "read_file", "description": "d", "parameters": {"type": "object"}}}
        flat = {"type": "function", "name": "read_file", "parameters": {"type": "object"}}
        result = self.convert(tools=[nested, flat])
        self.assertEqual(result["tools"], [{"type": "function", "name": "read_file", "description": "d",
                                            "parameters": {"type": "object"}}, flat])

    def test_a_response_format_becomes_the_text_format_engines_read(self):
        schema = {"name": "answer", "strict": True, "schema": {"type": "object"}}
        result = self.convert(response_format={"type": "json_schema", "json_schema": schema}, text={"verbosity": "low"})
        self.assertEqual(result["text"], {"verbosity": "low", "format": {"type": "json_schema", **schema}})
        self.assertNotIn("response_format", result)
        self.assertEqual(self.convert(response_format={"type": "json_object"})["text"], {"format": {"type": "json_object"}})

    def test_an_empty_response_format_leaves_the_text_format_alone(self):
        text = {"format": {"type": "json_object"}}
        self.assertEqual(self.convert(response_format=None, text=text), {"text": text})

    def test_other_fields_and_non_message_items_pass_through(self):
        body = {"model": "m", "stream": True, "input": [{"type": "function_call_output", "call_id": "c", "output": "x"}]}
        self.assertEqual(self.convert(**body), body)


class ForwardTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        StubClient.sent, StubClient.script = [], []
        self.request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/chat/completions"),
            headers={"authorization": "Bearer nativ", "content-type": "application/json", "host": "x",
                     "cookie": "session=1", "x-api-key": "k", "accept-encoding": "gzip"},
        )

    def forward(self, engine=ENGINE, payload=None):
        return asyncio.run(self.module.forward(self.request, payload or {"model": "qwen"}, engine))

    def collect(self, response):
        async def gather():
            return [chunk async for chunk in response.content]
        return asyncio.run(gather())

    def test_the_request_goes_to_the_engine_with_its_model_and_key(self):
        StubClient.script = [UpstreamResponse()]
        self.forward()
        sent = StubClient.sent[0]
        self.assertEqual(sent.url, "http://127.0.0.1:8091/v1/chat/completions")
        self.assertEqual(sent.json["model"], "engine-model")
        self.assertEqual(sent.headers["authorization"], "Bearer secret")

    def test_only_allowed_client_headers_reach_the_engine(self):
        StubClient.script = [UpstreamResponse()]
        self.forward(engine={"base_url": "http://127.0.0.1:8091/v1"})
        self.assertEqual(StubClient.sent[0].headers, {"content-type": "application/json"})

    def test_a_buffered_answer_drops_hop_headers_and_the_stale_content_encoding(self):
        upstream = UpstreamResponse()
        StubClient.script = [upstream]
        response = self.forward()
        self.assertEqual(response.content, b'{"ok": true}')
        self.assertEqual(response.headers, {"content-type": "application/json", "x-extra": "1"})
        self.assertEqual(upstream.closed, 1)

    def test_an_error_status_from_the_engine_is_passed_through(self):
        StubClient.script = [UpstreamResponse(body=b"bad", status_code=400, content_type="text/plain")]
        response = self.forward()
        self.assertEqual((response.status_code, response.content), (400, b"bad"))

    def test_a_failed_body_read_is_a_503_and_the_connection_is_closed(self):
        upstream = UpstreamResponse(read_error=StubHTTPError("reset"))
        StubClient.script = [upstream]
        self.assertEqual(self.forward().status_code, 503)
        self.assertEqual(upstream.closed, 1)

    def test_a_stream_is_relayed_chunk_by_chunk_then_closed(self):
        upstream = UpstreamResponse(content_type="text/event-stream", chunks=[b"data: 1\n\n", b"data: 2\n\n"])
        StubClient.script = [upstream]
        response = self.forward()
        self.assertEqual(self.collect(response), [b"data: 1\n\n", b"data: 2\n\n"])
        self.assertEqual(upstream.closed, 1)
        self.assertNotIn("content-length", response.headers)

    def responses_stream(self, chunks):
        self.request.url = SimpleNamespace(path="/v1/responses")
        upstream = UpstreamResponse(content_type="text/event-stream", chunks=chunks)
        StubClient.script = [upstream]
        return upstream, self.forward()

    def test_responses_timings_are_hoisted_to_the_event_the_app_reads(self):
        final = b'event: response.completed\ndata: ' + json.dumps(
            {"type": "response.completed", "response": {"timings": {"prompt_per_second": 207.0}}}).encode() + b"\n\n"
        upstream, response = self.responses_stream([b"event: a\ndata: {\"x\": 1}\n\n", final])
        events = b"".join(self.collect(response)).split(b"\n\n")
        hoisted = json.loads(events[1].split(b"data: ", 1)[1])
        self.assertEqual(hoisted["timings"], {"prompt_per_second": 207.0})
        self.assertEqual(hoisted["response"], {"timings": {"prompt_per_second": 207.0}})
        self.assertEqual(events[0], b'event: a\ndata: {"x": 1}')
        self.assertEqual(upstream.closed, 1)

    def test_a_responses_event_split_across_chunks_is_reassembled(self):
        final = b'data: {"response": {"timings": {"predicted_per_second": 90.0}}}\n\n'
        _, response = self.responses_stream([final[:20], final[20:]])
        out = b"".join(self.collect(response))
        self.assertEqual(json.loads(out.split(b"data: ", 1)[1])["timings"], {"predicted_per_second": 90.0})

    def test_responses_events_without_nested_timings_pass_through_unchanged(self):
        chunks = [b'data: {"delta": "hi"}\n\n', b"data: [DONE]\n\n", b"tail"]
        _, response = self.responses_stream(chunks)
        self.assertEqual(b"".join(self.collect(response)), b"".join(chunks))

    def test_timings_already_on_the_event_are_not_overwritten(self):
        event = b'data: {"timings": {"a": 1}, "response": {"timings": {"a": 2}}}\n\n'
        _, response = self.responses_stream([event])
        self.assertEqual(b"".join(self.collect(response)), event)

    def test_a_stream_that_breaks_midway_still_closes_the_connection(self):
        upstream = UpstreamResponse(content_type="text/event-stream", chunks=[b"data: 1\n\n", StubTransportError("reset")])
        StubClient.script = [upstream]
        response = self.forward()
        with self.assertRaises(StubTransportError):
            self.collect(response)
        self.assertEqual(upstream.closed, 1)

    def test_a_stale_pooled_connection_is_retried_once(self):
        StubClient.script = [StubTransportError("closed by peer"), UpstreamResponse()]
        self.assertEqual(self.forward().status_code, 200)
        self.assertEqual(len(StubClient.sent), 2)

    def test_an_engine_that_stays_down_is_a_503(self):
        StubClient.script = [StubTransportError("refused"), StubTransportError("refused")]
        self.assertEqual(self.forward().status_code, 503)

    def test_a_malformed_base_url_is_a_503_not_an_exception(self):
        StubClient.script = [StubInvalidURL("bad url")]
        self.assertEqual(self.forward().status_code, 503)
        self.assertEqual(len(StubClient.sent), 1)

    def test_only_responses_bodies_are_translated(self):
        StubClient.script = [UpstreamResponse(), UpstreamResponse()]
        messages = [{"role": "tool", "tool_call_id": "c1", "content": "x"}]
        self.forward(payload={"model": "qwen", "messages": messages})
        self.assertEqual(StubClient.sent[0].json["messages"], messages)
        self.request.url.path = "/v1/responses"
        self.forward(payload={"model": "qwen", "input": [{"type": "message", "role": "tool", "tool_call_id": "c1", "content": "x"}]})
        self.assertEqual(StubClient.sent[1].json["input"][0]["type"], "function_call_output")

    def test_compaction_is_a_501_and_never_reaches_the_engine(self):
        self.request.url.path = "/v1/responses/compact"
        self.assertEqual(self.forward().status_code, 501)
        self.assertEqual(StubClient.sent, [])


class StatusReply:
    """What urllib's urlopen returns: a context manager that json.load can read."""

    def __init__(self, body):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, *args):
        return self._body


METRICS = b"# HELP x\nsplash_memory_current_bytes 19751714816\nsplash_memory_peak_bytes 35179020288\n"
TIMED_EVENT = b'event: response.completed\ndata: {"response": {"timings": {"predicted_per_second": 70.0}}}'


class EngineMemoryTests(unittest.TestCase):
    ENGINES = {"qwen": {"base_url": "http://127.0.0.1:8091/v1"}}

    def setUp(self):
        self.module = load_module()

    def refresh(self, metrics):
        def reply(request, timeout=None):
            if request.full_url.endswith("/metrics"):
                if isinstance(metrics, Exception):
                    raise metrics
                return StatusReply(metrics)
            return StatusReply(b'{"ready": true, "maximum_context_tokens": 262144}')
        with patch.object(self.module.urllib.request, "urlopen", side_effect=reply):
            self.module.refresh_ready_engines(self.ENGINES)

    def hoisted_timings(self, event=TIMED_EVENT):
        data = json.loads(self.module._hoist_timings(event).split(b"data: ")[1])
        return data["timings"]

    def test_memory_in_use_is_read_in_gb_from_the_engine_metrics(self):
        self.refresh(METRICS)
        self.assertEqual(self.module.engine_memory_gb(), 19.75)

    def test_the_hoisted_timings_carry_it_as_peak_memory(self):
        self.refresh(METRICS)
        self.assertEqual(self.hoisted_timings(), {"predicted_per_second": 70.0, "peak_memory": 19.75})

    def test_an_engine_peak_already_in_the_timings_is_kept(self):
        self.refresh(METRICS)
        event = b'data: {"response": {"timings": {"peak_memory": 3.5}}}'
        self.assertEqual(self.hoisted_timings(event), {"peak_memory": 3.5})

    def test_without_a_memory_reading_the_timings_get_no_peak(self):
        for metrics in (OSError("no metrics"), b"nothing useful"):
            self.refresh(metrics)
            self.assertEqual(self.hoisted_timings(), {"predicted_per_second": 70.0})


class RemoteModelStatusTests(unittest.TestCase):
    ENGINES = {"qwen": {"base_url": "http://127.0.0.1:8091/v1"}}

    def setUp(self):
        self.module = load_module()

    def refresh(self, reply):
        with patch.object(self.module.urllib.request, "urlopen", side_effect=reply if isinstance(reply, Exception) else None,
                          return_value=None if isinstance(reply, Exception) else StatusReply(reply)) as opened:
            self.module.refresh_ready_engines(self.ENGINES)
        return opened

    def test_a_ready_engine_is_reported_as_the_loaded_model_with_its_context(self):
        self.refresh(b'{"ready": true, "maximum_context_tokens": 262144}')
        shown = self.module.with_remote_model_loaded({"loaded_model": None, "configured_context_limit": None})
        self.assertEqual(shown["loaded_model"], "qwen")
        self.assertEqual(shown["loaded_context_size"], 262144)
        self.assertEqual(shown["effective_context_limit"], 262144)

    def test_the_status_is_read_from_the_engine_root_not_its_v1_path(self):
        opened = self.refresh(b'{"ready": true}')
        self.assertEqual(opened.call_args_list[0].args[0].full_url, "http://127.0.0.1:8091/status")

    def test_a_configured_limit_below_the_engine_context_wins(self):
        self.refresh(b'{"ready": true, "maximum_context_tokens": 262144}')
        shown = self.module.with_remote_model_loaded({"loaded_model": None, "configured_context_limit": 65536})
        self.assertEqual(shown["effective_context_limit"], 65536)

    def test_a_model_loaded_in_process_is_never_replaced(self):
        self.refresh(b'{"ready": true, "maximum_context_tokens": 262144}')
        snapshot = {"loaded_model": "in-process", "effective_context_limit": 4096}
        self.assertIs(self.module.with_remote_model_loaded(snapshot), snapshot)

    def test_an_engine_that_is_not_ready_leaves_the_snapshot_alone(self):
        self.refresh(b'{"ready": false, "maximum_context_tokens": 262144}')
        snapshot = {"loaded_model": None}
        self.assertIs(self.module.with_remote_model_loaded(snapshot), snapshot)

    def test_an_unreachable_engine_leaves_the_snapshot_alone(self):
        self.refresh(OSError("connection refused"))
        snapshot = {"loaded_model": None}
        self.assertIs(self.module.with_remote_model_loaded(snapshot), snapshot)

    def test_a_garbled_status_leaves_the_snapshot_alone(self):
        self.refresh(b"<html>")
        snapshot = {"loaded_model": None}
        self.assertIs(self.module.with_remote_model_loaded(snapshot), snapshot)

    def test_an_engine_that_goes_away_is_forgotten_on_the_next_pass(self):
        self.refresh(b'{"ready": true}')
        self.refresh(OSError("gone"))
        snapshot = {"loaded_model": None}
        self.assertIs(self.module.with_remote_model_loaded(snapshot), snapshot)

    def test_the_reported_engine_model_is_also_reported_ready(self):
        self.refresh(b'{"ready": true}')
        shown = self.module.with_remote_model_loaded({"loaded_model": None, "model_ready": False,
                                                      "model_load_progress": None})
        self.assertTrue(shown["model_ready"])

    def test_nothing_is_claimed_while_a_model_is_loading_in_process(self):
        self.refresh(b'{"ready": true}')
        snapshot = {"loaded_model": None, "model_ready": False,
                    "model_load_progress": {"stage": "evaluating_layers"}}
        self.assertIs(self.module.with_remote_model_loaded(snapshot), snapshot)

    def test_a_load_that_failed_or_was_cancelled_does_not_hide_the_engine(self):
        self.refresh(b'{"ready": true}')
        for stage in ("failed", "cancelled"):
            with self.subTest(stage=stage):
                shown = self.module.with_remote_model_loaded({"loaded_model": None,
                                                              "model_load_progress": {"stage": stage}})
                self.assertEqual(shown["loaded_model"], "qwen")


class ExternalLoadGuardTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.listed = {"qwen": {"base_url": "http://127.0.0.1:8091/v1"}}
        self.module.listed_models = lambda *path: self.listed
        self.loaded = []
        self.guarded = self.module.refusing_external_loads(
            lambda *args, **kwargs: self.loaded.append((args, kwargs)) or "resources")

    def test_a_mapped_model_is_refused_before_the_loader_runs(self):
        with self.assertRaises(StubHTTPException) as refused:
            self.guarded("qwen", None, model_kind="text_generation")
        self.assertEqual(refused.exception.status_code, 501)
        self.assertIn("qwen", refused.exception.detail)
        self.assertEqual(self.loaded, [])

    def test_a_mapped_model_named_by_keyword_is_refused_too(self):
        with self.assertRaises(StubHTTPException):
            self.guarded(model_path="qwen")
        self.assertEqual(self.loaded, [])

    def test_a_listed_model_whose_entry_is_unusable_is_refused(self):
        self.listed = {"qwen": {"base-url": "http://127.0.0.1:8091/v1"}}
        with self.assertRaises(StubHTTPException):
            self.guarded("qwen")
        self.assertEqual(self.loaded, [])

    def test_nothing_loads_while_the_listed_models_are_unknown(self):
        self.listed = None
        with self.assertRaises(StubHTTPException) as refused:
            self.guarded("any-model")
        self.assertEqual(refused.exception.status_code, 503)
        self.assertIn("remote-engines.json", refused.exception.detail)
        self.assertEqual(self.loaded, [])

    def test_other_models_load_with_their_arguments_unchanged(self):
        self.assertEqual(self.guarded("local-model", "adapter", model_kind="auto"), "resources")
        self.assertEqual(self.loaded, [(("local-model", "adapter"), {"model_kind": "auto"})])

    def test_asking_to_load_a_mapped_model_succeeds_without_loading_it(self):
        self.assertEqual(self.module.external_load_reply("qwen", {}),
                         {"status": "loaded", "model": "qwen", "loaded_models": {}})

    def test_asking_to_load_any_other_model_is_left_to_the_server(self):
        self.assertIsNone(self.module.external_load_reply("local-model", {}))


class ForwardedSpeedAnalyticsTests(unittest.TestCase):
    """The server's analytics record the speeds and memory an external engine reports with a streamed reply."""

    TIMINGS = {"prompt_per_second": 223.7, "predicted_per_second": 44.7, "peak_memory": 19.46}

    def record(self, kind, *blocks):
        tree = ast.parse(SERVER_SOURCE.read_text())
        accumulator = next(node for node in tree.body
                           if isinstance(node, ast.ClassDef) and node.name == "StreamAccumulator")
        namespace = {"json": json, "time": time, "Any": object, "RequestObservation": object}
        exec(compile(ast.Module(body=[accumulator], type_ignores=[]), str(SERVER_SOURCE), "exec"), namespace)
        observation = SimpleNamespace(model="qwen", start_time=time.perf_counter(), first_token_at=None)
        stream = namespace["StreamAccumulator"](observation, kind)
        for block in blocks:
            stream.feed(block)
        recorded = stream.finalize()
        return {key: recorded[key] for key in ("prefill_tok_s", "decode_tok_s", "peak_memory_gb")}

    def completed(self, **extra):
        event = {"type": "response.completed", "response": {"usage": {"input_tokens": 337, "output_tokens": 69}}, **extra}
        return f"event: response.completed\ndata: {json.dumps(event)}"

    def test_a_responses_reply_records_the_engine_speeds_and_memory(self):
        self.assertEqual(self.record("responses", self.completed(timings=self.TIMINGS)),
                         {"prefill_tok_s": 223.7, "decode_tok_s": 44.7, "peak_memory_gb": 19.46})

    def test_a_responses_reply_without_timings_records_none(self):
        self.assertEqual(self.record("responses", self.completed()),
                         {"prefill_tok_s": None, "decode_tok_s": None, "peak_memory_gb": None})

    def test_a_zero_speed_does_not_replace_one_already_seen(self):
        earlier = 'event: response.output_text.done\ndata: {"timings": {"predicted_per_second": 50.0}}'
        recorded = self.record("responses", earlier, self.completed(timings={"predicted_per_second": 0}))
        self.assertEqual(recorded["decode_tok_s"], 50.0)

    def test_a_chat_reply_records_the_engine_prefill_speed(self):
        chunk = {"choices": [], "usage": {"prompt_tokens": 19, "completion_tokens": 2}, "timings": self.TIMINGS}
        self.assertEqual(self.record("chat", f"data: {json.dumps(chunk)}"),
                         {"prefill_tok_s": 223.7, "decode_tok_s": 44.7, "peak_memory_gb": 19.46})


class LoadGuardInstallTests(unittest.TestCase):
    """The server wraps whichever names the bundled server's callers really look up."""

    def install(self, app_names, package_names):
        tree = ast.parse(SERVER_SOURCE.read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "install_remote_engine_load_guard")
        app = SimpleNamespace(**{name: f"app.{name}" for name in app_names})
        package = SimpleNamespace(**{name: f"package.{name}" for name in package_names})
        namespace = {
            "base": package,
            "importlib": SimpleNamespace(import_module=lambda name: app),
            "remote_engine": SimpleNamespace(refusing_external_loads=lambda loader: ("guarded", loader)),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SERVER_SOURCE), "exec"), namespace)
        namespace["install_remote_engine_load_guard"]()
        return app, package

    def test_a_server_with_an_inner_loader_is_guarded_there_only(self):
        app, package = self.install(("get_cached_model", "_get_cached_model"), ("get_cached_model",))
        self.assertEqual(app._get_cached_model, ("guarded", "app._get_cached_model"))
        self.assertEqual(app.get_cached_model, "app.get_cached_model")
        self.assertEqual(package.get_cached_model, "package.get_cached_model")

    def test_an_older_server_is_guarded_in_the_module_and_the_package_copy(self):
        app, package = self.install(("get_cached_model",), ("get_cached_model",))
        self.assertEqual(app.get_cached_model, ("guarded", "app.get_cached_model"))
        self.assertEqual(package.get_cached_model, ("guarded", "package.get_cached_model"))

    def test_a_server_with_no_known_loader_stops_the_launch(self):
        with self.assertRaises(AttributeError):
            self.install((), ())


def summary_reply(text="SUM"):
    return JSONReply({"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]})


def history():
    """A user task, a long early tool exchange, then a short recent exchange."""
    return [
        {"type": "message", "role": "user", "content": "task"},
        {"type": "function_call", "call_id": "c1", "name": "read_file", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "x" * 5000},
        {"type": "message", "role": "assistant", "content": "noted"},
        {"type": "message", "role": "user", "content": "next"},
        {"type": "function_call", "call_id": "c2", "name": "read_file", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c2", "output": "short"},
    ]


def chat_history():
    """Long text-only turns: there is no tool output to clear, so shrinking it needs a summary."""
    turns = []
    for number in range(4):
        turns.append({"type": "message", "role": "user", "content": f"question {number} " + "q" * 3000})
        turns.append({"type": "message", "role": "assistant", "content": f"answer {number} " + "a" * 3000})
    return turns + [{"type": "message", "role": "user", "content": "latest"}]


def completed_event(tokens, output=None):
    data = {"type": "response.completed", "response": {"output": output or [], "usage": {"input_tokens": tokens}}}
    return b"event: response.completed\ndata: " + json.dumps(data).encode()


class CompactionTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        StubClient.posted, StubClient.post_script = [], []

    def compact(self, tokens=1000, threshold=500, body=None, items=None):
        body = body or {"model": "m", "input": items or chat_history(), "tools": [], "tool_choice": "auto", "stream": True}
        return asyncio.run(self.module.compaction_item(
            StubClient(), "http://engine/v1/responses", {}, body, threshold, {"usage": {"input_tokens": tokens}}))

    def test_the_threshold_is_read_from_the_apps_context_management(self):
        read = self.module.compaction_threshold
        self.assertEqual(read({"context_management": [{"type": "compaction", "compact_threshold": 196608}]}), 196608)
        for payload in ({}, {"context_management": []}, {"context_management": "x"},
                        {"context_management": [{"type": "other", "compact_threshold": 5}]},
                        {"context_management": [{"type": "compaction", "compact_threshold": 0}]},
                        {"context_management": [{"type": "compaction", "compact_threshold": "9"}]}):
            self.assertIsNone(read(payload))

    def test_a_sealed_item_expands_back_to_its_context_and_replaces_what_came_before(self):
        carried = [{"type": "message", "role": "user", "content": "kept"}]
        item = self.module.seal_compaction(carried)
        later = {"type": "message", "role": "user", "content": "later"}
        expanded = self.module.expand_compaction([{"type": "message", "role": "user", "content": "old"}, item, later])
        self.assertEqual(expanded, carried + [later])

    def test_input_without_a_compaction_item_is_unchanged(self):
        items = history()
        self.assertEqual(self.module.expand_compaction(items), items)

    def test_a_foreign_or_damaged_compaction_item_becomes_a_note_not_an_error(self):
        later = {"type": "message", "role": "user", "content": "later"}
        for content in ("opaque-from-another-server", self.module.CAPSULE_PREFIX + "{not json",
                        self.module.CAPSULE_PREFIX + '{"items": 5}', None):
            expanded = self.module.expand_compaction([{"type": "compaction", "encrypted_content": content}, later])
            self.assertEqual(len(expanded), 2)
            self.assertEqual(expanded[-1], later)
            self.assertIn("compacted", expanded[0]["content"])

    def test_the_cut_keeps_the_final_group_and_as_many_earlier_ones_as_fit(self):
        items = history()
        self.assertEqual(self.module.choose_cut(items, 125, 0.185), 4)
        self.assertEqual(self.module.choose_cut(items, 0, 0.185), 5)

    def test_a_conversation_with_nothing_before_its_final_group_is_not_cut(self):
        self.assertEqual(self.module.choose_cut(history()[:1], 125, 0.185), 0)

    def test_the_compacted_context_is_anchors_then_handoff_then_the_tail(self):
        items = history()
        compacted = self.module.compacted_items(items, " the summary ", 4)
        self.assertEqual(compacted[0], items[0])
        self.assertIn("the summary", compacted[1]["content"][0]["text"])
        self.assertEqual(compacted[2:], items[4:])

    def test_the_first_and_latest_user_messages_before_the_cut_are_both_kept_once(self):
        items = history()
        self.assertEqual(self.module.compacted_items(items, "s", 5)[:2], [items[0], items[4]])
        self.assertEqual(self.module.compacted_items(items, "s", 2)[0], items[0])
        self.assertEqual(len([i for i in self.module.compacted_items(items, "s", 2) if i is items[0]]), 1)

    def test_below_the_threshold_nothing_is_compacted_and_the_engine_is_not_asked(self):
        self.assertIsNone(self.compact(tokens=100))
        self.assertEqual(StubClient.posted, [])

    def test_at_the_threshold_the_prompt_just_answered_plus_one_instruction_is_sent(self):
        StubClient.post_script = [summary_reply()]
        self.assertIsNotNone(self.compact(tokens=500))
        sent = StubClient.posted[0].json
        self.assertEqual(sent["input"][:-1], chat_history())
        self.assertEqual(sent["input"][-1]["content"], self.module.SUMMARY_INSTRUCTION)
        self.assertIs(sent["stream"], False)
        self.assertNotIn("tool_choice", sent)

    def test_the_sealed_result_expands_to_the_task_the_summary_and_the_recent_tail(self):
        StubClient.post_script = [summary_reply("THE HANDOFF")]
        expanded = self.module.expand_compaction([self.compact()])
        source = chat_history()
        handoff = next(i for i, item in enumerate(expanded) if isinstance(item["content"], list))
        self.assertIn("THE HANDOFF", expanded[handoff]["content"][0]["text"])
        self.assertEqual(expanded[0], source[0])
        self.assertEqual(expanded[handoff + 1:], source[-1:])
        self.assertLess(len(expanded), len(source))

    def test_an_empty_summary_or_an_engine_error_leaves_the_conversation_alone(self):
        for step in (summary_reply("  "), JSONReply({}, error=StubHTTPError("down")), JSONReply({"output": []})):
            StubClient.post_script = [step]
            self.assertIsNone(self.compact())

    def test_a_summary_that_would_not_shrink_the_context_is_discarded(self):
        StubClient.post_script = [summary_reply("y" * 50000)]
        self.assertIsNone(self.compact())

    def test_old_tool_output_is_cleared_first_without_asking_the_engine(self):
        item = self.compact(items=history())
        cleared = self.module.expand_compaction([item])
        self.assertEqual(StubClient.posted, [])
        self.assertEqual(cleared[2]["output"], self.module.CLEARED_OUTPUT)
        self.assertEqual(cleared[6], history()[6])
        self.assertEqual([i.get("content") for i in cleared if i.get("type") == "message"],
                         [i.get("content") for i in history() if i.get("type") == "message"])

    def test_clearing_keeps_every_call_and_the_recent_tail_verbatim(self):
        items = history()
        cleared = self.module.clear_old_outputs(items, 4)
        self.assertEqual([i["call_id"] for i in cleared if i["type"] == "function_call"], ["c1", "c2"])
        self.assertEqual(cleared[4:], items[4:])
        self.assertEqual(cleared[2]["output"], self.module.CLEARED_OUTPUT)

    def test_when_clearing_is_not_enough_the_engine_is_asked_for_a_summary(self):
        StubClient.post_script = [summary_reply()]
        self.assertIsNotNone(self.compact())
        self.assertEqual(len(StubClient.posted), 1)

    def test_only_the_completed_event_gets_the_item(self):
        async def compact(response):
            return {"type": "compaction", "id": "cmp_1"}
        add = lambda event, fn=compact: asyncio.run(self.module._with_compaction(event, fn))
        output = json.loads(add(completed_event(9, [{"type": "message"}])).split(b"data: ")[1])["response"]["output"]
        self.assertEqual([item["type"] for item in output], ["message", "compaction"])
        delta = b'event: response.output_text.delta\ndata: {"type": "response.output_text.delta", "delta": "hi"}'
        self.assertEqual(add(delta), delta)
        self.assertEqual(add(completed_event(9), None), completed_event(9))

    def test_a_declined_compaction_leaves_the_final_event_byte_identical(self):
        async def decline(response):
            return None
        event = completed_event(9)
        self.assertEqual(asyncio.run(self.module._with_compaction(event, decline)), event)


class PrefillProgressTests(unittest.TestCase):
    """The app draws its "Reading prompt" ring from log lines the bundled engine prints; the engine's counters feed the same lines."""

    def setUp(self):
        self.module = load_module()
        self.lines = []

    def run_reporter(self, counters, total=20000, minimum=2048):
        script = list(counters)

        async def scenario():
            exhausted = asyncio.Event()

            async def read_counters():
                if script:
                    return script.pop(0)
                exhausted.set()
                await asyncio.sleep(3600)

            async def count_tokens():
                return total

            reporter = self.module.PrefillReporter(
                "req1", count_tokens, read_counters, self.lines.append, interval=0, minimum=minimum
            )
            task = asyncio.create_task(reporter.run())
            waiting = asyncio.create_task(exhausted.wait())
            await asyncio.wait({task, waiting}, return_when=asyncio.FIRST_COMPLETED)
            waiting.cancel()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            return reporter
        return asyncio.run(scenario())

    def test_a_cold_prompt_reports_start_then_growing_progress_then_completion(self):
        reporter = self.run_reporter([(1000, 50), (1000, 50), (6000, 50), (12000, 50), (21000, 50)])
        reporter.finish()
        self.assertEqual(self.lines[0], "Prefill started: request=req1 prompt_tokens=20000")
        progress = [int(line.split("tokens=")[1].split("/")[0]) for line in self.lines if line.startswith("Prefill progress")]
        self.assertEqual(progress, sorted(progress))
        self.assertEqual(progress[-1], 20000)
        self.assertTrue(all(line.endswith("/20000") for line in self.lines if line.startswith("Prefill progress")))
        self.assertEqual(self.lines[-1], "Prefill completed: request=req1 prompt_tokens=20000")

    def test_cached_tokens_count_as_already_read(self):
        reporter = self.run_reporter([(0, 0), (0, 15000), (1000, 15000)])
        first = next(line for line in self.lines if line.startswith("Prefill progress"))
        self.assertEqual(first, "Prefill progress: request=req1 tokens=15000/20000")

    def test_a_mostly_cached_prompt_shows_nothing(self):
        reporter = self.run_reporter([(0, 0), (0, 19500), (200, 19500)])
        reporter.finish()
        self.assertEqual(self.lines, [])

    def test_progress_never_exceeds_the_prompt_size(self):
        self.run_reporter([(0, 0), (50000, 0)])
        self.assertTrue(all(int(line.split("tokens=")[1].split("/")[0]) <= 20000
                            for line in self.lines if line.startswith("Prefill progress")))

    def test_unreadable_counters_show_nothing_and_do_not_raise(self):
        reporter = self.run_reporter([None, None, None])
        reporter.finish()
        self.assertEqual(self.lines, [])

    def test_a_failing_counter_read_ends_the_poller_without_raising(self):
        async def broken():
            raise RuntimeError("boom")

        async def count():
            return 20000

        async def scenario():
            reporter = self.module.PrefillReporter("req1", count, broken, self.lines.append, interval=0)
            with self.assertLogs(level="ERROR"):
                await reporter.run()
        asyncio.run(scenario())
        self.assertEqual(self.lines, [])

    def test_finish_is_idempotent(self):
        reporter = self.run_reporter([(0, 0), (9000, 0)])
        reporter.finish()
        reporter.finish()
        self.assertEqual(sum(line.startswith("Prefill completed") for line in self.lines), 1)

    def test_the_counters_are_read_from_the_engines_metrics_text(self):
        text = "splash_scheduler_prefill_rows_total 4930035\nsplash_cache_reused_tokens_total 14985728\nother 1\n"
        self.assertEqual(self.module.parse_prefill_counters(text), (4930035, 14985728))
        self.assertIsNone(self.module.parse_prefill_counters("nothing here"))


class PrefillStreamScopeTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def run_stream(self, chunks):
        finished = []

        async def scenario():
            async def relay():
                for chunk in chunks:
                    yield chunk

            async def forever():
                await asyncio.sleep(3600)

            task = asyncio.create_task(forever())
            reporter = SimpleNamespace(finish=lambda: finished.append(1))
            out = [c async for c in self.module._with_prefill_report(relay(), reporter, task)]
            await asyncio.sleep(0)
            return out, task.cancelled(), len(finished)
        return asyncio.run(scenario())

    def test_the_ring_closes_when_the_first_generated_text_arrives(self):
        chunks = [b'data: {"type": "response.created"}\n\n', b'data: {"type": "response.output_text.delta"}\n\n']
        out, cancelled, finishes = self.run_stream(chunks)
        self.assertEqual(out, chunks)
        self.assertTrue(cancelled)
        self.assertGreaterEqual(finishes, 1)

    def test_the_ring_closes_when_a_stream_ends_without_text(self):
        out, cancelled, finishes = self.run_stream([b'data: {"type": "response.created"}\n\n'])
        self.assertTrue(cancelled)
        self.assertGreaterEqual(finishes, 1)


class PrefillTaskLifecycleTests(unittest.TestCase):
    """The progress poller must stop whenever the request ends, including when the engine cannot be reached."""

    def setUp(self):
        self.module = load_module()
        StubClient.sent, StubClient.script, StubClient.posted, StubClient.post_script = [], [], [], []

    def forward_and_count_tasks(self, script):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        StubClient.script = script
        StubClient.post_script = [JSONReply({"prompt": "p"}), JSONReply({"tokens": list(range(20000))})]
        StubClient.metrics_text = "splash_scheduler_prefill_rows_total 0\nsplash_cache_reused_tokens_total 0\n"

        async def scenario():
            await self.module.forward(request, {"model": "qwen", "stream": True, "input": chat_history()}, ENGINE)
            await asyncio.sleep(0)
            return [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
        return asyncio.run(scenario())

    def test_no_poller_is_left_running_when_the_engine_is_unreachable(self):
        pending = self.forward_and_count_tasks([StubTransportError("refused"), StubTransportError("refused")])
        self.assertEqual(pending, [])

    def test_no_poller_is_left_running_for_a_reply_that_is_not_a_stream(self):
        pending = self.forward_and_count_tasks([UpstreamResponse(content_type="application/json", chunks=[b"{}"])])
        self.assertEqual(pending, [])


class RequestStabilityTests(unittest.TestCase):
    """The engine's prefix cache matches raw text, so the same tools must always be serialised the same way."""

    def setUp(self):
        self.module = load_module()
        StubClient.sent, StubClient.script, StubClient.posted, StubClient.post_script = [], [], [], []

    def sent_tools_text(self, path, tools):
        request = SimpleNamespace(url=SimpleNamespace(path=path), headers={})
        StubClient.script = [UpstreamResponse(content_type="text/event-stream", chunks=[b"data: [DONE]\n\n"])]
        payload = {"model": "qwen", "stream": True, "tools": tools,
                   "input": chat_history(), "messages": [{"role": "user", "content": "hi"}]}
        response = asyncio.run(self.module.forward(request, payload, ENGINE))

        async def drain():
            return [chunk async for chunk in response.content]
        asyncio.run(drain())
        return json.dumps(StubClient.sent[-1].json["tools"])

    def test_the_same_tools_in_a_different_key_order_are_sent_identically(self):
        first = [{"type": "function", "function": {"name": "a", "parameters": {"type": "object", "required": ["x"],
                  "properties": {"x": {"type": "string", "description": "d"}}}}}]
        second = [{"function": {"parameters": {"properties": {"x": {"description": "d", "type": "string"}},
                   "required": ["x"], "type": "object"}, "name": "a"}, "type": "function"}]
        for path in ("/v1/responses", "/v1/chat/completions"):
            with self.subTest(path=path):
                self.assertEqual(self.sent_tools_text(path, first), self.sent_tools_text(path, second))

    def test_the_order_of_the_tool_list_and_of_required_fields_is_kept(self):
        tools = [{"type": "function", "function": {"name": "b", "parameters": {"required": ["z", "a"]}}},
                 {"type": "function", "function": {"name": "a", "parameters": {"required": ["m"]}}}]
        sent = json.loads(self.sent_tools_text("/v1/chat/completions", tools))
        self.assertEqual([t["function"]["name"] for t in sent], ["b", "a"])
        self.assertEqual(sent[0]["function"]["parameters"]["required"], ["z", "a"])


class CompactionForwardTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        StubClient.sent, StubClient.script, StubClient.posted, StubClient.post_script = [], [], [], []
        self.request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})

    def payload(self, **extra):
        return {"model": "qwen", "stream": True, "input": chat_history(),
                "context_management": [{"type": "compaction", "compact_threshold": 500}], **extra}

    def forward(self, payload):
        return asyncio.run(self.module.forward(self.request, payload, ENGINE))

    def collect(self, response):
        async def gather():
            return [chunk async for chunk in response.content]
        return b"".join(asyncio.run(gather()))

    def test_the_apps_compaction_request_is_not_passed_to_the_engine(self):
        StubClient.script = [UpstreamResponse(content_type="text/event-stream", chunks=[b"data: [DONE]\n\n"])]
        self.collect(self.forward(self.payload()))
        self.assertNotIn("context_management", StubClient.sent[0].json)

    def test_a_stream_past_the_threshold_ends_with_a_compaction_item(self):
        StubClient.script = [UpstreamResponse(content_type="text/event-stream",
                                              chunks=[completed_event(1000) + b"\n\n"])]
        StubClient.post_script = [summary_reply()]
        body = self.collect(self.forward(self.payload()))
        output = json.loads(body.split(b"data: ")[1])["response"]["output"]
        self.assertEqual(len(StubClient.posted), 1)
        self.assertEqual(output[-1]["type"], "compaction")
        self.assertTrue(output[-1]["encrypted_content"].startswith(self.module.CAPSULE_PREFIX))

    def test_a_stream_under_the_threshold_is_relayed_unchanged_and_never_summarised(self):
        event = completed_event(100) + b"\n\n"
        StubClient.script = [UpstreamResponse(content_type="text/event-stream", chunks=[event])]
        self.assertEqual(self.collect(self.forward(self.payload())), event)
        self.assertEqual(StubClient.posted, [])

    def test_a_buffered_answer_past_the_threshold_gets_the_item_too(self):
        answer = {"output": [], "usage": {"input_tokens": 1000}}
        StubClient.script = [UpstreamResponse(body=json.dumps(answer).encode())]
        StubClient.post_script = [summary_reply()]
        response = self.forward(self.payload(stream=False))
        self.assertEqual(len(StubClient.posted), 1)
        self.assertEqual(json.loads(response.content)["output"][-1]["type"], "compaction")

    def test_a_stored_compaction_item_is_expanded_before_the_engine_sees_the_input(self):
        item = self.module.seal_compaction([{"type": "message", "role": "user", "content": "carried"}])
        newer = {"type": "message", "role": "user", "content": "newer"}
        StubClient.script = [UpstreamResponse()]
        self.forward(self.payload(stream=False, input=[item, newer], context_management=None))
        self.assertEqual([i["content"] for i in StubClient.sent[0].json["input"]], ["carried", "newer"])

    def test_a_compaction_item_from_elsewhere_does_not_stick_the_chat(self):
        bad = {"type": "compaction", "encrypted_content": "from-another-server"}
        newer = {"type": "message", "role": "user", "content": "newer"}
        StubClient.script = [UpstreamResponse()]
        self.forward(self.payload(stream=False, input=[bad, newer], context_management=None))
        self.assertEqual(StubClient.sent[0].json["input"][-1]["content"], "newer")


def excerpts(*notes, max_bytes=4000):
    return {"max_bytes": max_bytes, "entries": [{"tool_call_id": call, "text": text} for call, text in notes]}


class RequestContextTests(unittest.TestCase):
    """Nativ sends two request-local fields that its bundled server writes into the prompt. Other engines
    ignore them, so the forwarder has to place their text itself."""

    def setUp(self):
        self.module = load_module()
        StubClient.sent, StubClient.script, StubClient.posted, StubClient.post_script = [], [], [], []

    def sent(self, payload, path="/v1/responses"):
        request = SimpleNamespace(url=SimpleNamespace(path=path), headers={})
        StubClient.script, StubClient.post_script = [UpstreamResponse()], [JSONReply({})]
        asyncio.run(self.module.forward(request, {"model": "qwen", **payload}, ENGINE))
        return StubClient.sent[-1].json

    def test_the_current_environment_text_is_the_last_message_the_engine_sees(self):
        body = self.sent({"input": history(), "transient_context": "DOCUMENT EXCERPT"})
        self.assertEqual(body["input"][-1], {"type": "message", "role": "user", "content": "DOCUMENT EXCERPT"})
        self.assertEqual(len(body["input"]), len(history()) + 1)
        self.assertNotIn("transient_context", body)

    def test_a_chat_completions_request_gets_the_environment_text_as_a_chat_message(self):
        messages = [{"role": "user", "content": "hi"}]
        body = self.sent({"messages": messages, "transient_context": "NOW"}, "/v1/chat/completions")
        self.assertEqual(body["messages"], messages + [{"role": "user", "content": "NOW"}])
        self.assertNotIn("transient_context", body)

    def test_a_request_with_a_plain_string_input_keeps_it_ahead_of_the_environment_text(self):
        body = self.sent({"input": "question", "transient_context": "NOW"})
        self.assertEqual([item["content"] for item in body["input"]], ["question", "NOW"])

    def test_excerpts_are_restored_ahead_of_the_history_only_for_results_no_longer_in_it(self):
        body = self.sent({"input": history(), "tool_result_context": excerpts(("gone", "LOST FACT"), ("c2", "STILL THERE"))})
        first = body["input"][0]
        self.assertEqual(first["role"], "user")
        self.assertTrue(first["content"].startswith(self.module.RECOVERY_HEADING))
        self.assertIn("LOST FACT", first["content"])
        self.assertNotIn("STILL THERE", first["content"])
        self.assertNotIn("tool_result_context", body)

    def test_a_result_that_compaction_cleared_counts_as_no_longer_in_the_history(self):
        cleared = self.module.clear_old_outputs(history(), len(history()))
        body = self.sent({"input": cleared, "tool_result_context": excerpts(("c1", "CLEARED FACT"))})
        self.assertIn("CLEARED FACT", body["input"][0]["content"])

    def test_chat_excerpts_follow_the_system_message_and_skip_results_still_present(self):
        messages = [{"role": "system", "content": "rules"}, {"role": "tool", "tool_call_id": "c9", "content": "seen"},
                    {"role": "user", "content": "hi"}]
        body = self.sent({"messages": messages, "tool_result_context": excerpts(("gone", "LOST"), ("c9", "SEEN"))},
                         "/v1/chat/completions")
        self.assertEqual(body["messages"][0], messages[0])
        self.assertIn("LOST", body["messages"][1]["content"])
        self.assertNotIn("SEEN", body["messages"][1]["content"])
        self.assertEqual(body["messages"][2:], messages[1:])

    def test_excerpts_are_trimmed_and_then_dropped_to_stay_inside_the_apps_byte_budget(self):
        notes = [(f"gone{number}", f"note{number} " + "z" * 2000) for number in range(12)]
        text = self.sent({"input": history(), "tool_result_context": excerpts(*notes, max_bytes=600)})["input"][0]["content"]
        self.assertLessEqual(len(text.encode()), 600)
        self.assertIn("note11 ", text)  # the newest excerpt is kept and the oldest dropped, as the bundled server does
        self.assertNotIn("note0 ", text)

    def test_excerpts_for_results_all_still_present_add_nothing(self):
        body = self.sent({"input": history(), "tool_result_context": excerpts(("c1", "A"), ("c2", "B"))})
        self.assertEqual(len(body["input"]), len(history()))

    def test_a_malformed_context_field_is_dropped_without_failing_the_request(self):
        for field in ("text", {"entries": "x"}, {"entries": [None, {"tool_call_id": "a"}]}, 5):
            with self.subTest(field=field):
                body = self.sent({"input": history(), "tool_result_context": field, "transient_context": {"not": "text"}})
                self.assertEqual(len(body["input"]), len(history()))

    def test_the_token_count_includes_both_blocks_where_generation_puts_them(self):
        StubClient.post_script = [JSONReply({"prompt": "p"}), JSONReply({"tokens": [1]})]
        payload = {"model": "qwen", "input": [{"type": "message", "role": "user", "content": "hi"}],
                   "transient_context": "NOW", "tool_result_context": excerpts(("gone", "LOST"))}
        asyncio.run(self.module.count_input_tokens(payload, ENGINE))
        contents = [message["content"] for message in StubClient.posted[0].json["messages"]]
        self.assertIn("LOST", contents[0])
        self.assertEqual(contents[1:], ["hi", "NOW"])

    def test_neither_block_is_sealed_into_a_compaction_item_and_the_summary_keeps_the_cached_prefix(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        StubClient.script = [UpstreamResponse(body=json.dumps({"output": [], "usage": {"input_tokens": 1000}}).encode())]
        StubClient.post_script = [JSONReply({}), summary_reply()]
        payload = {"model": "qwen", "input": chat_history(), "transient_context": "NOW STATE",
                   "tool_result_context": excerpts(("gone", "LOST FACT")),
                   "context_management": [{"type": "compaction", "compact_threshold": 500}]}
        response = asyncio.run(self.module.forward(request, payload, ENGINE))
        sent, summary = StubClient.sent[-1].json["input"], StubClient.posted[-1].json["input"]
        self.assertEqual(summary[:-1], sent[:-1])  # everything but the current-state block is the prompt just answered
        self.assertEqual(summary[:-1], StubClient.posted[0].json["input"])  # and what the warm-up left cached
        self.assertNotIn("NOW STATE", json.dumps(summary))
        capsule = json.dumps(self.module.expand_compaction(json.loads(response.content)["output"][-1:]))
        self.assertNotIn("NOW STATE", capsule)
        self.assertNotIn("LOST FACT", capsule)


class WarmUpTests(unittest.TestCase):
    """The engine keeps reusable state only at the end of a prompt. With a block trailing the history, that point
    lies past where the next turn diverges, so the history is first processed on its own."""

    def setUp(self):
        self.module = load_module()
        StubClient.sent, StubClient.script, StubClient.posted, StubClient.post_script = [], [], [], []

    def forward(self, payload, path="/v1/responses", warm_up=None):
        request = SimpleNamespace(url=SimpleNamespace(path=path), headers={"content-type": "application/json"})
        StubClient.script, StubClient.post_script = [UpstreamResponse()], [warm_up or JSONReply({})]
        return asyncio.run(self.module.forward(request, {"model": "qwen", **payload}, ENGINE))

    def test_the_history_alone_is_processed_before_a_request_with_a_trailing_block(self):
        sent_when_warmed = []

        class Reply(JSONReply):
            def raise_for_status(self):
                sent_when_warmed.append(len(StubClient.sent))

        self.forward({"input": history(), "transient_context": "NOW", "stream": True, "store": True,
                      "return_progress": True, "max_output_tokens": 900, "tools": [{"type": "function", "name": "f"}],
                      "tool_result_context": excerpts(("gone", "LOST"))}, warm_up=Reply({}))
        (warm,), (real,) = StubClient.posted, StubClient.sent
        self.assertEqual(sent_when_warmed, [0])
        self.assertEqual((warm.url, warm.headers), (real.url, real.headers))
        self.assertEqual(warm.json["input"], real.json["input"][:-1])
        self.assertEqual((warm.json["stream"], warm.json["store"], warm.json["max_output_tokens"]), (False, False, 1))
        self.assertNotIn("return_progress", warm.json)
        for field in ("model", "tools", "chat_template_kwargs"):
            self.assertEqual(warm.json.get(field), real.json.get(field))
        self.assertEqual(real.json["max_output_tokens"], 900)

    def test_a_chat_completions_warm_up_caps_whichever_output_limit_the_client_named(self):
        for limit in ("max_tokens", "max_completion_tokens"):
            with self.subTest(limit=limit):
                StubClient.sent, StubClient.posted = [], []
                self.forward({"messages": [{"role": "user", "content": "hi"}], "transient_context": "NOW", limit: 900,
                              "stream": True, "stream_options": {"include_usage": True}}, "/v1/chat/completions")
                warm = StubClient.posted[0].json
                self.assertEqual(warm[limit], 1)
                self.assertEqual(warm["messages"], [{"role": "user", "content": "hi"}])
                self.assertNotIn("stream_options", warm)
                self.assertNotIn("store", warm)

    def test_nothing_extra_is_sent_when_no_block_trails_the_history(self):
        for extra in ({}, {"tool_result_context": excerpts(("gone", "LOST"))}, {"transient_context": ""}):
            with self.subTest(extra=extra):
                StubClient.posted = []
                self.forward({"input": history(), **extra})
                self.assertEqual(StubClient.posted, [])

    def test_a_failed_warm_up_never_costs_the_request(self):
        for failure in (JSONReply({}, error=StubHTTPError("busy")), StubHTTPError("refused"), StubInvalidURL("bad")):
            with self.subTest(failure=failure):
                StubClient.sent = []
                response = self.forward({"input": history(), "transient_context": "NOW"}, warm_up=failure)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(len(StubClient.sent), 1)


ANSWER = b'data: {"type": "response.output_text.delta", "delta": "hi"}\n\n'


class SlowWarmUpTests(unittest.TestCase):
    """A long read must not leave the app waiting in silence, or carrying on after it has gone."""

    def setUp(self):
        self.module = load_module()
        self.module.WARM_UP_WAIT_S = 0.01
        StubClient.sent, StubClient.script, StubClient.posted, StubClient.post_script = [], [], [], []
        StubClient.metrics_text = ""  # another test's counters would start the progress reporter, which also posts
        self.request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        self.payload = {"model": "qwen", "stream": True, "input": history(), "transient_context": "NOW"}

    async def start(self):
        """forward() must answer promptly however long the engine takes; a test that waits longer has failed."""
        return await asyncio.wait_for(self.module.forward(self.request, self.payload, ENGINE), 1)

    def run_slow(self, upstream, chunks_before_release=3):
        """Start a request whose warm-up hangs, take some chunks, release it, and take the rest."""
        pending = Pending()
        StubClient.post_script, StubClient.script = [pending], [upstream]

        async def scenario():
            response = await self.start()
            stream = response.body_iterator
            early = [await asyncio.wait_for(anext(stream), 1) for _ in range(chunks_before_release)]
            sent_early = len(StubClient.sent)
            pending.release.set()
            return response, early, sent_early, b"".join([chunk async for chunk in stream])
        return asyncio.run(scenario())

    def test_the_app_gets_keepalives_while_the_engine_is_still_reading(self):
        upstream = UpstreamResponse(content_type="text/event-stream", chunks=[ANSWER])
        response, early, sent_early, rest = self.run_slow(upstream)
        self.assertEqual((response.status_code, response.media_type), (200, "text/event-stream"))
        self.assertEqual(early, [self.module.KEEPALIVE] * 3)
        self.assertTrue(self.module.KEEPALIVE.startswith(b":") and self.module.KEEPALIVE.endswith(b"\n\n"))
        self.assertEqual(sent_early, 0)
        self.assertIn(ANSWER, rest)

    def test_an_engine_error_after_a_slow_warm_up_reaches_the_app_as_a_stream_error(self):
        upstream = UpstreamResponse(body=b'{"error": {"message": "prompt is too long"}}', status_code=400)
        *_, rest = self.run_slow(upstream)
        event = json.loads(rest.split(b"data: ")[-1])
        self.assertEqual(event["type"], "error")
        self.assertIn("prompt is too long", event["error"]["message"])
        self.assertIn("400", event["error"]["message"])

    def test_an_unreachable_engine_after_a_slow_warm_up_reaches_the_app_as_a_stream_error(self):
        pending = Pending()
        StubClient.post_script, StubClient.script = [pending], [StubTransportError("down"), StubTransportError("down")]

        async def scenario():
            response = await self.start()
            pending.release.set()
            return b"".join([chunk async for chunk in response.body_iterator])
        event = json.loads(asyncio.run(scenario()).split(b"data: ")[-1])
        self.assertEqual(event["type"], "error")
        self.assertIn("down", event["error"]["message"])

    def test_leaving_during_a_slow_warm_up_cancels_it_and_sends_nothing_more(self):
        pending = Pending()
        StubClient.post_script = [pending]

        async def scenario():
            response = await self.start()
            await anext(response.body_iterator)
            await response.body_iterator.aclose()
            for _ in range(3):
                await asyncio.sleep(0)
            return len([task for task in asyncio.all_tasks() if task is not asyncio.current_task()])
        left_running = asyncio.run(scenario())
        self.assertTrue(pending.cancelled)
        self.assertEqual(StubClient.sent, [])
        self.assertEqual(left_running, 0)

    def test_closing_the_reply_closes_what_it_wraps_straight_away(self):
        closed = []

        async def inner():
            try:
                yield b"a"
                yield b"b"
            finally:
                closed.append(True)

        async def scenario():
            held = inner()  # a live reference, so only an explicit close can run its cleanup
            wrapper = self.module._with_prefill_report(held, SimpleNamespace(finish=lambda: None),
                                                       asyncio.create_task(asyncio.sleep(0)))
            await anext(wrapper)
            await wrapper.aclose()
            return list(closed)
        self.assertEqual(asyncio.run(scenario()), [True])

    def test_cancelling_the_request_during_the_first_wait_leaves_no_task_behind(self):
        self.module.WARM_UP_WAIT_S = 60
        pending = Pending()
        StubClient.post_script = [pending]
        # Counters the progress reporter can read, so it is still polling when the request is cancelled.
        StubClient.metrics_text = "splash_scheduler_prefill_rows_total 0\nsplash_cache_reused_tokens_total 0\n"
        self.addCleanup(setattr, StubClient, "metrics_text", "")

        async def count_tokens(payload, engine):
            return 50000
        self.module._count_prompt_tokens = count_tokens

        async def scenario():
            request = asyncio.create_task(self.module.forward(self.request, self.payload, ENGINE))
            await asyncio.sleep(0.02)
            request.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await request
            for _ in range(3):
                await asyncio.sleep(0)
            return len([task for task in asyncio.all_tasks() if task is not asyncio.current_task()])
        left_running = asyncio.run(scenario())
        self.assertTrue(pending.cancelled)
        self.assertEqual(left_running, 0)

    def test_an_unexpected_warm_up_failure_is_logged_and_the_request_still_goes(self):
        StubClient.post_script, StubClient.script = [RuntimeError("boom")], [UpstreamResponse()]
        self.payload["stream"] = False
        with self.assertLogs("nativ.remote_engine", "WARNING") as logs:
            response = asyncio.run(self.start())
        self.assertEqual((response.status_code, len(StubClient.sent)), (200, 1))
        self.assertIn("boom", "".join(logs.output))

    def test_a_warm_up_on_a_stale_connection_is_tried_again(self):
        StubClient.post_script = [StubTransportError("stale"), JSONReply({})]
        StubClient.script = [UpstreamResponse()]
        self.payload["stream"] = False
        asyncio.run(self.start())
        self.assertEqual(len(StubClient.posted), 2)


class ThinkingSettingsTests(unittest.TestCase):
    """The app names its thinking settings at the top level; engines read them elsewhere and ignore those names."""

    def setUp(self):
        self.module = load_module()
        StubClient.sent, StubClient.script, StubClient.posted, StubClient.post_script = [], [], [], []

    def sent(self, payload, path):
        request = SimpleNamespace(url=SimpleNamespace(path=path), headers={})
        StubClient.script = [UpstreamResponse()]
        conversation = {"input": history(), "messages": [{"role": "user", "content": "hi"}]}
        asyncio.run(self.module.forward(request, {"model": "qwen", **conversation, **payload}, ENGINE))
        return StubClient.sent[-1].json

    def test_the_thinking_switch_reaches_the_chat_template_on_both_routes(self):
        for path in ("/v1/responses", "/v1/chat/completions"):
            for enabled in (True, False):
                with self.subTest(path=path, enabled=enabled):
                    body = self.sent({"enable_thinking": enabled}, path)
                    self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": enabled})
                    self.assertNotIn("enable_thinking", body)

    def test_the_apps_switch_wins_and_other_template_arguments_are_kept(self):
        body = self.sent({"enable_thinking": False,
                          "chat_template_kwargs": {"enable_thinking": True, "custom": 1}}, "/v1/responses")
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False, "custom": 1})

    def test_the_thinking_budget_is_sent_under_the_engines_name(self):
        body = self.sent({"thinking_budget": 512}, "/v1/responses")
        self.assertEqual(body["thinking_token_budget"], 512)
        self.assertNotIn("thinking_budget", body)

    def test_a_budget_the_client_already_named_for_the_engine_is_kept(self):
        body = self.sent({"thinking_budget": 512, "thinking_token_budget": 64}, "/v1/chat/completions")
        self.assertEqual(body["thinking_token_budget"], 64)

    def test_a_request_without_thinking_settings_gains_none(self):
        body = self.sent({"enable_thinking": None}, "/v1/responses")
        for field in ("chat_template_kwargs", "thinking_token_budget", "enable_thinking"):
            self.assertNotIn(field, body)


class CompactionRobustnessTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        StubClient.posted, StubClient.post_script = [], []

    def compaction_item(self, body, tokens=1000, threshold=500):
        return asyncio.run(self.module.compaction_item(
            StubClient(), "http://engine/v1/responses", {}, body, threshold, {"usage": {"input_tokens": tokens}}))

    def test_tool_calls_and_outputs_kept_in_a_capsule_are_counted_as_chat_messages(self):
        carried = [{"type": "message", "role": "user", "content": "task"},
                   {"type": "function_call", "call_id": "c1", "name": "read_file", "arguments": '{"path": "a"}'},
                   {"type": "function_call_output", "call_id": "c1", "output": "BODY"}]
        payload = {"model": "qwen", "input": [self.module.seal_compaction(carried),
                                              {"type": "message", "role": "user", "content": "hi"}]}
        StubClient.post_script = [JSONReply({"prompt": "p"}), JSONReply({"tokens": [1]})]
        asyncio.run(self.module.count_input_tokens(payload, ENGINE))
        messages = StubClient.posted[0].json["messages"]
        self.assertTrue(all(m.get("role") for m in messages))
        call = next(m for m in messages if m.get("tool_calls"))
        self.assertEqual(call["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(next(m for m in messages if m["role"] == "tool")["content"], "BODY")

    def test_a_compaction_that_raises_never_costs_the_reply(self):
        async def broken(response):
            raise RuntimeError("boom")
        event = completed_event(9)
        self.assertEqual(asyncio.run(self.module._with_compaction(event, broken)), event)

    def test_odd_requests_decline_instead_of_raising(self):
        for body in ({"model": "m"}, {"model": "m", "input": ["x", {"type": "function_call"}]}):
            StubClient.post_script = [summary_reply()]
            self.assertIsNone(self.compaction_item(body))
        StubClient.post_script = [JSONReply({"output": [{"type": "message", "content": None}]})]
        self.assertIsNone(self.compaction_item({"model": "m", "input": chat_history()}))

    def test_a_chat_stored_with_a_foreign_capsule_can_still_be_counted(self):
        foreign = {"type": "compaction", "encrypted_content": "from-another-server"}
        StubClient.post_script = [JSONReply({"prompt": "p"}), JSONReply({"tokens": [1, 2]})]
        reply = asyncio.run(self.module.count_input_tokens({"model": "qwen", "input": [foreign]}, ENGINE))
        self.assertEqual(reply.content, {"input_tokens": 2})

    def test_a_truncated_summary_is_not_sealed(self):
        StubClient.post_script = [JSONReply({"status": "incomplete", "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "half a handoff"}]}]})]
        self.assertIsNone(self.compaction_item({"model": "m", "input": chat_history()}))

    def test_the_summary_request_drops_output_format_fields_and_stays_inside_the_threshold(self):
        StubClient.post_script = [summary_reply()]
        body = {"model": "m", "input": chat_history(), "text": {"format": {"type": "json_schema"}},
                "response_format": {"type": "json_object"}}
        self.compaction_item(body)
        sent = StubClient.posted[0].json
        self.assertNotIn("text", sent)
        self.assertNotIn("response_format", sent)
        self.assertLessEqual(sent["max_output_tokens"], 50)

    def test_a_cut_never_separates_a_call_from_its_output_even_when_calls_are_not_adjacent(self):
        items = [{"type": "message", "role": "user", "content": "go"},
                 {"type": "function_call", "call_id": "a", "name": "f", "arguments": "{}"},
                 {"type": "message", "role": "assistant", "content": "thinking"},
                 {"type": "function_call", "call_id": "b", "name": "f", "arguments": "{}"},
                 {"type": "function_call_output", "call_id": "a", "output": "A" * 40},
                 {"type": "function_call_output", "call_id": "b", "output": "B" * 40},
                 {"type": "message", "role": "user", "content": "next"},
                 {"type": "message", "role": "assistant", "content": "ok"}]
        for budget in (0, len(json.dumps(items[3:])) + 1, 10 ** 6):
            tail = items[self.module.choose_cut(items, budget, 1.0):]
            calls = {i["call_id"] for i in tail if i.get("type") == "function_call"}
            self.assertTrue(all(i["call_id"] in calls for i in tail if i.get("type") == "function_call_output"))

    def test_the_final_event_gets_compaction_even_without_a_trailing_blank_line(self):
        async def compact(response):
            return {"type": "compaction", "id": "cmp_1"}
        upstream = UpstreamResponse(content_type="text/event-stream", chunks=[completed_event(9)])

        async def gather():
            return [chunk async for chunk in self.module._relay_responses_and_close(upstream, compact)]
        self.assertIn(b'"compaction"', b"".join(asyncio.run(gather())))

    def test_timings_that_are_not_a_mapping_leave_the_event_unchanged(self):
        event = b'data: {"response": {"timings": null}}'
        self.assertEqual(self.module._hoist_timings(event), event)

    def summary_request(self, body, app_request=None, tokens=1000, threshold=500):
        StubClient.posted, StubClient.post_script = [], [summary_reply()]
        asyncio.run(self.module.compaction_item(StubClient(), "http://engine/v1/responses", {}, body, threshold,
                                                {"usage": {"input_tokens": tokens}}, app_request=app_request))
        return StubClient.posted[0].json if StubClient.posted else None

    def test_the_summary_generates_no_reasoning_and_leaves_the_prompts_rendering_alone(self):
        rendering = {"chat_template_kwargs": {"enable_thinking": True, "custom": 1}, "reasoning": {"effort": "high"}}
        body = {"model": "m", "input": chat_history(), "temperature": 0.8, "thinking_token_budget": 512, **rendering}
        sent = self.summary_request(body)
        self.assertEqual((sent["thinking_token_budget"], sent["temperature"]), (0, 0))
        # Anything the engine renders into the prompt must match the chat's request, or its cache misses.
        self.assertEqual({key: sent[key] for key in rendering}, rendering)
        self.assertEqual(body["thinking_token_budget"], 512)  # the chat's own request is untouched

    def test_the_summary_is_told_to_leave_out_recovered_excerpts_only_when_there_are_some(self):
        body = {"model": "m", "input": chat_history()}
        plain = self.summary_request(body)["input"][-1]["content"]
        recovered = self.summary_request(body, {"tool_result_context": excerpts(("gone", "X"))})["input"][-1]["content"]
        self.assertEqual(plain, self.module.SUMMARY_INSTRUCTION)
        self.assertEqual(recovered, self.module.SUMMARY_INSTRUCTION + "\n" + self.module.SUMMARY_SKIPS_RECOVERY)

    def test_a_large_current_state_block_stops_clearing_alone_counting_as_enough(self):
        items = history()
        block = "z" * 3000
        for state, summarised in ((None, False), (block, True)):
            with self.subTest(block=len(state or "")):
                size = self.module._text_size(self.module._with_context(items, (None, state), True))
                sent = self.summary_request({"model": "m", "input": items}, {"transient_context": state},
                                            tokens=size, threshold=4000)
                self.assertEqual(sent is not None, summarised)

    def test_clearing_is_judged_with_the_excerpts_the_cleared_history_will_bring_back(self):
        items = history()
        cleared_size = self.module._text_size(self.module.clear_old_outputs(items, 4))
        threshold = int((cleared_size + 200) / self.module.CLEAR_TARGET)  # fits cleared, but not with an excerpt back
        size = self.module._text_size(items)  # one token per character
        for note, summarised in (("gone-before-and-after", False), ("c1", True)):
            with self.subTest(excerpt_for=note):
                # "c1" is in the history now, so it costs nothing until clearing blanks it.
                app_request = {"tool_result_context": excerpts((note, "E" * 2000))} if note == "c1" else None
                sent = self.summary_request({"model": "m", "input": items}, app_request, tokens=size, threshold=threshold)
                self.assertEqual(sent is not None, summarised)

    def test_a_second_compaction_replaces_the_first_handoff_instead_of_stacking_it(self):
        StubClient.post_script = [summary_reply("FIRST")]
        first = self.compaction_item({"model": "m", "input": chat_history()})
        expanded = self.module.expand_compaction([first] + chat_history())
        StubClient.post_script = [summary_reply("SECOND")]
        second = self.compaction_item({"model": "m", "input": expanded})
        carried = self.module.expand_compaction([second])
        handoffs = [i for i in carried if isinstance(i.get("content"), list) and "handoff" in json.dumps(i["content"]).lower()]
        self.assertEqual(len(handoffs), 1)
        self.assertIn("SECOND", json.dumps(handoffs[0]))
        self.assertIn("FIRST", json.dumps(StubClient.posted[1].json["input"]))


class EngineStatusRobustnessTests(unittest.TestCase):
    ENGINES = {"a": {"base_url": "http://a:1/v1"}, "b": {"base_url": "http://b:1/v1"}}

    def setUp(self):
        self.module = load_module()

    def refresh(self, metrics_by_host):
        def reply(request, timeout=None):
            host = request.full_url.split("//")[1].split(":")[0]
            if request.full_url.endswith("/metrics"):
                return StatusReply(metrics_by_host[host])
            return StatusReply(b'{"ready": true}')
        with patch.object(self.module.urllib.request, "urlopen", side_effect=reply):
            self.module.refresh_ready_engines(self.ENGINES)

    def test_a_malformed_http_reply_counts_as_not_ready_and_does_not_raise(self):
        # the module under test's own http.client: the loader restores sys.modules, so a fresh import would be a different class
        with patch.object(self.module.urllib.request, "urlopen", side_effect=self.module.http.client.BadStatusLine("x")):
            self.module.refresh_ready_engines(self.ENGINES)
        self.assertIsNone(self.module.with_remote_model_loaded({"loaded_model": None}).get("loaded_model"))

    def test_memory_in_float_or_exponent_form_is_read(self):
        for text, expected in ((b"splash_memory_current_bytes 1.7e+10\n", 17.0), (b"splash_memory_current_bytes 123000000000.0\n", 123.0)):
            self.refresh({"a": text, "b": text})
            self.assertEqual(self.module.engine_memory_gb("a"), expected)

    def test_the_memory_reported_is_that_of_the_engine_serving_the_request(self):
        self.refresh({"a": b"splash_memory_current_bytes 10000000000\n", "b": b"splash_memory_current_bytes 20000000000\n"})
        event = b'data: {"response": {"timings": {"predicted_per_second": 1}}}'
        for model, expected in (("a", 10.0), ("b", 20.0)):
            timings = json.loads(self.module._hoist_timings(event, model).split(b"data: ")[1])["timings"]
            self.assertEqual(timings["peak_memory"], expected)

    def test_the_refresh_loop_survives_an_unexpected_error(self):
        with patch.object(self.module, "refresh_ready_engines", side_effect=RuntimeError("boom")) as refresh:
            self.module.refresh_once()
        self.assertEqual(refresh.call_count, 1)


if __name__ == "__main__":
    unittest.main()
