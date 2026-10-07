"""Checks the server without calling ElevenLabs (the network calls are faked)."""
import io

import pytest

import app as A

N = len(A.DEFAULT_SENTENCES)   # sentences you record
E = len(A.DEFAULT_EXTRA)       # new sentences, AI only


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
    posts, deleted = [], []

    def fake_post(url, **kwargs):
        posts.append(url)
        if url.endswith("/voices/add"):
            return Resp({"voice_id": f"v{sum(u.endswith('/voices/add') for u in posts)}"})   # v1, v2, ...
        return Resp()

    monkeypatch.setattr(A, "DATA", tmp_path)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    monkeypatch.setattr(A.requests, "post", fake_post)
    monkeypatch.setattr(A.requests, "delete", lambda url, **k: deleted.append(url) or Resp())
    return type("Env", (), {"client": staticmethod(A.app.test_client), "posts": posts, "deleted": deleted})


def upload(c, i):
    return c.post(f"/api/real/{i}", content_type="multipart/form-data",
                  data={"audio": (io.BytesIO(b"x"), "r.wav")})


def record(c, n=N):
    for i in range(n):
        upload(c, i)


def full_run(c):
    record(c)
    assert c.post("/api/clone").status_code == 200
    assert c.post("/api/fakes").status_code == 200


def test_visitors_are_kept_apart(env):
    a, b = env.client(), env.client()
    record(a)
    record(b, n=1)
    assert a.get("/api/state").json["has_real"] == [True] * N
    assert b.get("/api/state").json["has_real"] == [True] + [False] * (N - 1)


def test_recorded_sentences_get_four_clips_and_new_ones_get_three(env):
    c = env.client()
    full_run(c)
    assert c.get("/api/state").json["ready"]
    assert sorted(c.get("/api/answer/0").json["order"]) == ["alt1", "alt2", "clone", "real"]
    assert sorted(c.get(f"/api/answer/{N}").json["order"]) == ["alt1", "alt2", "clone"]   # first new sentence
    assert all(c.get(f"/clip/0/{slot}").status_code == 200 for slot in range(4))
    assert all(c.get(f"/clip/{N}/{slot}").status_code == 200 for slot in range(3))
    assert c.get(f"/clip/{N}/3").status_code == 404                                          # no real clip


def test_three_different_clones_speak_and_extras_are_deleted(env):
    full_run(env.client())
    speech = [u for u in env.posts if "text-to-speech" in u]
    assert len(speech) == 3 * (N + E)                                  # 3 AI clips per sentence
    assert {u.split("/")[-1] for u in speech} == {"v1", "v2", "v3"}    # main clone + two partial clones
    assert sorted(u.split("/")[-1] for u in env.deleted) == ["v2", "v3"]   # temporary ones removed, main kept


def test_limits(env):
    c = env.client()
    assert c.post("/api/sentences", json={"sentences": ["x"] * 9}).status_code == 400
    assert c.post("/api/sentences", json={"sentences": ["x"], "extra": ["y"] * 5}).status_code == 400
    record(c)
    assert c.post("/api/clone").status_code == 200
    assert c.post("/api/clone").status_code == 400          # one clone at a time
    assert upload(c, 99).status_code == 404


def test_editing_only_the_new_sentences_keeps_recordings(env):
    c = env.client()
    full_run(c)
    sentences = c.get("/api/state").json["sentences"]
    assert c.post("/api/sentences", json={"sentences": sentences, "extra": ["A brand new line."]}).status_code == 200
    s = c.get("/api/state").json
    assert s["has_real"] == [True] * N and s["extra"] == ["A brand new line."] and not s["ready"]


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
