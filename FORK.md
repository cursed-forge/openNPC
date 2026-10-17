# CursedForge fork of openNPC

This fork is maintained by the CursedForge project (`cursed-forge` on GitHub).
Upstream: https://github.com/balaraj74/openNPC (MIT). Base commit:
`05e6be96c1053f31bd4083daca12b3ea2b6fde14`.

Branch `cursedforge` adds:

1. `opennpc/remote_llm.py`: `RemoteLLMEngine`, an engine that calls an
   OpenAI-compatible chat endpoint (Open-WebUI, llama.cpp) with the
   standard library only. The model id is pinned. `load` refuses a server
   that does not list the pinned id, and `generate` refuses an answer from
   another model. It passes `response_format` (strict JSON schemas) and
   extra body fields through.
2. `opennpc/llm.py`: `trust_remote_code=False` for the local engine, and
   `create_engine("local" | "remote", ...)`.
3. `tests/test_remote_llm.py`: tests against a local stub server
   (`python -m unittest tests.test_remote_llm`).

The CursedForge NPC gateway installs this branch at a pinned commit and
uses only the core package (no optional extras).
