"""Checks the server without calling ElevenLabs (the network calls are faked)."""
import io

import pytest

import app as A

VOICES = {"voices": [
    {"voice_id": f"{g[0].upper()}{n}", "category": "premade", "labels": {"gender": g}}
    for g in ("female", "male") for n in (1, 2)
]}


class Resp:
    ok = True
    status_code = 200
    content = b"mp3"
    text = ""

    def __init__(self, data=None):
        self.data = data or {"voice_id": "v1"}

    def json(self):
        return self.data


@pytest.fixture
def env(tmp_path, monkeypatch):
    posts = []

    def fake_post(url, **kwargs):
        posts.append(url)
        return Resp()

    monkeypatch.setattr(A, "DATA", tmp_path)
    monkeypatch.setattr(A, "_stock", {})
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    monkeypatch.setattr(A.requests, "post", fake_post)
    monkeypatch.setattr(A.requests, "get", lambda *a, **k: Resp(VOICES))
    monkeypatch.setattr(A.requests, "delete", lambda *a, **k: Resp())
    return type("Env", (), {"client": staticmethod(A.app.test_client), "posts": posts})


def record(c, n=5, pitch=120):
    for i in range(n):
        c.post(f"/api/real/{i}", content_type="multipart/form-data",
               data={"audio": (io.BytesIO(b"x"), "r.wav"), "pitch": str(pitch)})


def full_run(c, pitch=120):
    record(c, pitch=pitch)
    assert c.post("/api/clone").status_code == 200
    assert c.post("/api/fakes").status_code == 200


def test_visitors_are_kept_apart(env):
    a, b = env.client(), env.client()
    record(a)
    record(b, n=1)
    assert a.get("/api/state").json["has_real"] == [True] * 5
    assert b.get("/api/state").json["has_real"] == [True] + [False] * 4


def test_full_flow_gives_four_shuffled_clips(env):
    c = env.client()
    full_run(c)
    assert c.get("/api/state").json["ready"]
    assert sorted(c.get("/api/answer/0").json["order"]) == ["clone", "real", "stock1", "stock2"]
    assert all(c.get(f"/clip/0/{slot}").status_code == 200 for slot in range(4))
    assert c.get("/clip/0/4").status_code == 404


def test_stock_voices_match_the_speaker(env):
    full_run(env.client(), pitch=215)
    assert any("F1" in u for u in env.posts) and not any("M1" in u for u in env.posts)
    env.posts.clear()
    full_run(env.client(), pitch=115)
    assert any("M1" in u for u in env.posts) and not any("F1" in u for u in env.posts)


def test_limits(env):
    c = env.client()
    assert c.post("/api/sentences", json={"sentences": ["x"] * 9}).status_code == 400
    record(c)
    assert c.post("/api/clone").status_code == 200
    assert c.post("/api/clone").status_code == 400          # one clone at a time
    assert c.post("/api/real/99", content_type="multipart/form-data",
                  data={"audio": (io.BytesIO(b"x"), "r.wav")}).status_code == 404


def test_forged_cookie_is_ignored(env, tmp_path):
    c = env.client()
    r = c.get("/api/state", headers={"Cookie": "vtt_sid=../../etc"})
    assert r.status_code == 200
    assert not (tmp_path.parent / "etc").exists()


def test_delete_buttons(env):
    c = env.client()
    full_run(c)
    c.post("/api/delete-recordings")
    s = c.get("/api/state").json
    assert not s["ready"] and not any(s["has_real"]) and s["has_voice"]
    c.post("/api/delete-clone")
    assert not c.get("/api/state").json["has_voice"]
