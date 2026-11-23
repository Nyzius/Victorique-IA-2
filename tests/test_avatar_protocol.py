import json

import pytest

from app.config import settings
from app.core.avatar_protocol import (
    AVATAR_INSTRUCTIONS,
    EXPRESSIONS,
    GESTURES,
    AvatarControlParser,
    avatar_manifest,
)
from app.core.assistant import assistant_engine
from app.memory.conversation_store import conversation_store
from app.models.ollama_client import ollama_client


def test_registered_controls_survive_every_stream_split():
    raw = "[expression:skeptical]\r\n[gesture:page-turn]\nOra, isto é evidente."
    for split in range(len(raw) + 1):
        parser = AvatarControlParser()
        controls = []
        visible = []
        for chunk in (raw[:split], raw[split:]):
            found, text = parser.feed(chunk)
            controls.extend(found)
            visible.extend(text)
        found, text = parser.finish()
        controls.extend(found)
        visible.extend(text)
        assert controls == [
            {"type": "avatar", "expression": "skeptical"},
            {"type": "avatar", "gesture": "page-turn"},
        ]
        assert "".join(visible) == "Ora, isto é evidente."


def test_unknown_leading_controls_are_hidden_but_nonleading_text_is_preserved():
    parser = AvatarControlParser()
    controls, text = parser.feed("[expression:invented]\nResposta")
    assert controls == []
    assert "".join(text) == "Resposta"

    parser = AvatarControlParser()
    controls, text = parser.feed("Resposta [expression:happy]")
    assert controls == []
    assert "".join(text) == "Resposta [expression:happy]"


def test_prompt_and_ids_are_bounded_data():
    assert "[expression:ID]" in AVATAR_INSTRUCTIONS
    assert "[gesture:ID]" in AVATAR_INSTRUCTIONS
    assert len(EXPRESSIONS) == len(set(EXPRESSIONS))
    assert len(GESTURES) == len(set(GESTURES))
    assert all(value.replace("-", "").isalnum() for value in (*EXPRESSIONS, *GESTURES))


def test_avatar_manifest_is_the_media_free_renderer_contract(client):
    manifest = avatar_manifest()
    assert manifest["schemaVersion"] == 1
    assert manifest["renderer"] == "procedural-css"
    assert manifest["media"] is False
    assert manifest["expressions"] == list(EXPRESSIONS)
    assert [item["id"] for item in manifest["gestures"]] == list(GESTURES)

    response = client.get("/api/avatar/manifest")
    assert response.status_code == 200
    assert response.json() == manifest


@pytest.mark.asyncio
async def test_model_controls_become_sse_and_never_reach_saved_text(monkeypatch):
    async def fake_tokens(*args, **kwargs):
        for token in ("[expression:happy]", "\n[gesture:tea]", "\nHmph. Resolvido."):
            yield token

    async def fake_context(*args, **kwargs):
        return {"system_prompt": "teste", "messages": [], "sources": []}

    monkeypatch.setattr(ollama_client, "chat_stream", fake_tokens)
    monkeypatch.setattr("app.core.assistant.context_builder.build_context", fake_context)
    monkeypatch.setattr(settings, "tool_calling_enabled", False)

    conv = conversation_store.create_conversation(title="Avatar protocol")
    try:
        events = []
        async for chunk in assistant_engine.chat_stream(
            conversation_id=conv["id"], user_message="resolva", backend="ollama"
        ):
            events.append(json.loads(chunk.removeprefix("data: ")))

        avatar = [event for event in events if event["type"] == "avatar"]
        assert avatar == [
            {"type": "avatar", "expression": "happy"},
            {"type": "avatar", "gesture": "tea"},
        ]
        assert "".join(event["content"] for event in events if event["type"] == "token") == "Hmph. Resolvido."
        assert events[-1]["content"] == "Hmph. Resolvido."
        saved = conversation_store.get_messages(conv["id"])[-1]["content"]
        assert saved == "Hmph. Resolvido."
    finally:
        conversation_store.delete_conversation(conv["id"])
