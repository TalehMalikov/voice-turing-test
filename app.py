"""Voice Turing test: can your class tell your real voice from an ElevenLabs clone?

Every visitor gets a private workspace, tied to a cookie in their browser.
Run locally:  python app.py   then open http://localhost:5000
Run for real: waitress-serve --port=8000 app:app   (behind HTTPS, see README)
"""
import json
import os
import random
import re
import secrets
from pathlib import Path

import requests
from flask import Flask, abort, g, jsonify, request, send_file
from werkzeug.middleware.proxy_fix import ProxyFix

ROOT = Path(__file__).parent
DATA = ROOT / "voices"      # one folder per visitor, named like voices/calm-otter-3f2a9c1b7d4e8a05/
API = "https://api.elevenlabs.io/v1"
MODEL = "eleven_multilingual_v2"

# Limits, because the ElevenLabs account is shared by every visitor.
MAX_SENTENCES = 8
MAX_SENTENCE_CHARS = 200
MAX_GENERATIONS = 3                                   # per visitor
MAX_CLONES = int(os.environ.get("MAX_CLONES", 8))     # clones alive at once, all visitors

COOKIE = "vtt_sid"
SID_RE = re.compile(r"^[a-z]{3,10}-[a-z]{3,10}-[0-9a-f]{16}$")
ADJECTIVES = ["calm", "bright", "quiet", "brave", "gentle", "swift", "sunny", "clever", "mellow", "lucky"]
ANIMALS = ["otter", "heron", "fox", "lynx", "finch", "panda", "koala", "gecko", "badger", "robin"]

MAX_EXTRA = 4                                         # new sentences (AI only) per visitor

DEFAULT_SENTENCES = [
    "Honestly, I think the second option is a little better, but I would want to hear what everyone else thinks first.",
    "Could you please send me the notes from yesterday's lecture, and let me know if anything changes before Friday?",
    "I am running a little late, so go ahead and start without me, and I will catch up as soon as I can.",
]
DEFAULT_EXTRA = [
    "It was a quiet morning, and nothing seemed out of place until the phone started ringing.",
    "I will be there in ten minutes, so please save me a seat near the back.",
]

FAKES = ["clone", "alt1", "alt2"]

app = Flask(__name__, static_folder=None)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1)       # so HTTPS is detected behind a proxy
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


def new_session_id():
    """A friendly name plus a long random part, e.g. calm-otter-3f2a9c1b7d4e8a05."""
    return f"{random.choice(ADJECTIVES)}-{random.choice(ANIMALS)}-{secrets.token_hex(8)}"


def load_env():
    """Read ELEVENLABS_API_KEY from a .env file if there is one."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


load_env()


def api_key():
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        abort(400, "ELEVENLABS_API_KEY is not set on the server.")
    return key


# ---- one private folder per visitor --------------------------------------
@app.before_request
def identify_visitor():
    sid = request.cookies.get(COOKIE, "")
    g.new_sid = None
    if not SID_RE.match(sid):          # also blocks anything that isn't a plain id
        sid = new_session_id()
        g.new_sid = sid
    g.sid = sid
    g.dir = DATA / sid


@app.after_request
def remember_visitor(resp):
    if g.get("new_sid"):
        resp.set_cookie(COOKIE, g.new_sid, max_age=60 * 60 * 24 * 30,
                        httponly=True, samesite="Lax", secure=request.is_secure)
    return resp


def session_name():
    """The friendly part only (calm-otter). Safe to show; the random part stays secret."""
    return g.sid.rsplit("-", 1)[0]


def load_state():
    s = {"sentences": DEFAULT_SENTENCES, "extra": DEFAULT_EXTRA, "voice_id": None, "order": {}, "generations": 0}
    state_file = g.dir / "state.json"
    if state_file.exists():
        s.update(json.loads(state_file.read_text()))
    return s


def save_state(s):
    g.dir.mkdir(parents=True, exist_ok=True)
    (g.dir / "state.json").write_text(json.dumps(s, indent=2))


def real_path(i):
    return g.dir / f"real_{i}.wav"


def clip_path(kind, i):
    return real_path(i) if kind == "real" else g.dir / f"{kind}_{i}.mp3"


def active_clones():
    n = 0
    for f in DATA.glob("*/state.json"):
        try:
            n += bool(json.loads(f.read_text()).get("voice_id"))
        except ValueError:
            pass
    return n


def check(resp):
    """Turn an ElevenLabs error into a readable message."""
    if not resp.ok:
        print("ElevenLabs error:", resp.status_code, resp.text[:500], flush=True)
        abort(502, f"ElevenLabs said {resp.status_code}: {resp.text[:300]}")
    return resp


@app.errorhandler(400)
@app.errorhandler(413)
@app.errorhandler(502)
def on_error(e):
    return jsonify(error=e.description), e.code


# ---- pages and state -----------------------------------------------------
@app.get("/")
def index():
    return send_file(ROOT / "index.html")


@app.get("/api/state")
def state():
    s = load_state()
    n = len(s["sentences"])
    rounds = n + len(s["extra"])        # recorded sentences first, then the new ones
    has_real = [real_path(i).exists() for i in range(n)]
    has_fake = [all(clip_path(k, i).exists() for k in FAKES) for i in range(rounds)]
    ready = all(has_real) and all(has_fake) and all(str(i) in s["order"] for i in range(rounds))
    return jsonify(
        sentences=s["sentences"],
        extra=s["extra"],
        has_real=has_real,
        has_fake=has_fake,
        ready=ready,
        session=session_name(),
        has_voice=bool(s["voice_id"]),
        has_key=bool(os.environ.get("ELEVENLABS_API_KEY")),
    )


@app.post("/api/sentences")
def set_sentences():
    clean = lambda items: [x.strip() for x in items if x.strip()]
    sentences = clean(request.json["sentences"])
    extra = clean(request.json.get("extra", []))
    if not sentences:
        abort(400, "Need at least one sentence to record.")
    if len(sentences) > MAX_SENTENCES:
        abort(400, f"Record at most {MAX_SENTENCES} sentences.")
    if len(extra) > MAX_EXTRA:
        abort(400, f"Use at most {MAX_EXTRA} new sentences.")
    if any(len(x) > MAX_SENTENCE_CHARS for x in sentences + extra):
        abort(400, f"Keep each sentence under {MAX_SENTENCE_CHARS} characters.")
    s = load_state()
    if sentences != s["sentences"]:                  # recordings only match the old text
        for f in g.dir.glob("real_*.wav"):
            f.unlink()
    for f in g.dir.glob("*_*.mp3"):                  # generated clips have to be redone
        f.unlink()
    s.update(sentences=sentences, extra=extra, order={})
    save_state(s)
    return jsonify(ok=True)


@app.post("/api/real/<int:i>")
def upload_real(i):
    s = load_state()
    if i >= len(s["sentences"]):
        abort(404)
    g.dir.mkdir(parents=True, exist_ok=True)
    request.files["audio"].save(real_path(i))
    save_state(s)                       # also refreshes the folder's "last used" time
    return jsonify(ok=True)


# ---- ElevenLabs steps ----------------------------------------------------
@app.post("/api/clone")
def clone():
    key = api_key()
    s = load_state()
    if s["voice_id"]:
        abort(400, "You already have a clone. Delete it first to make a new one.")
    files = sorted(g.dir.glob("real_*.wav"))
    if not files:
        abort(400, "Record at least one sentence first.")
    if active_clones() >= MAX_CLONES:
        abort(400, "The demo is full right now. Please try again later.")

    s["voice_id"] = create_voice(key, f"turing-test-{session_name()}", files)
    save_state(s)
    return jsonify(ok=True)


def create_voice(key, name, files):
    """Upload recordings to ElevenLabs and return the new voice's id."""
    handles = [open(f, "rb") for f in files]
    try:
        resp = requests.post(
            f"{API}/voices/add",
            headers={"xi-api-key": key},
            data={"name": name},
            files=[("files", (f.name, h, "audio/wav")) for f, h in zip(files, handles)],
            timeout=120,
        )
    except requests.RequestException as e:
        print("Upload failed:", repr(e), flush=True)
        abort(502, "The upload to ElevenLabs was cut off. Please try again.")
    finally:
        for h in handles:
            h.close()
    return check(resp).json()["voice_id"]


@app.post("/api/fakes")
def make_fakes():
    key = api_key()
    s = load_state()
    if not s["voice_id"]:
        abort(400, "Clone your voice first.")
    if s["generations"] >= MAX_GENERATIONS:
        abort(400, f"You can generate clips {MAX_GENERATIONS} times per visit. Delete your clone to start over.")
    n = len(s["sentences"])
    files = [real_path(i) for i in range(n)]
    if not all(f.exists() for f in files):
        abort(400, "Record all the sentences first.")

    # Two extra clones, each hearing only part of your recordings (with 1 recording, all of it).
    subsets = {"alt1": files[:-1] or files, "alt2": files[1:] or files}
    temp = {}
    try:
        for kind, subset in subsets.items():
            temp[kind] = create_voice(key, f"turing-test-{session_name()}-{kind}", subset)
        voice_for = {"clone": s["voice_id"], **temp}
        for i, text in enumerate(s["sentences"] + s["extra"]):
            for kind in FAKES:
                resp = requests.post(
                    f"{API}/text-to-speech/{voice_for[kind]}",
                    headers={"xi-api-key": key},
                    json={"text": text, "model_id": MODEL},
                    timeout=120,
                )
                clip_path(kind, i).write_bytes(check(resp).content)
            order = (["real"] if i < n else []) + FAKES      # new sentences have no real recording
            random.shuffle(order)
            s["order"][str(i)] = order      # order[0] is clip A, order[1] is B, ...
    finally:
        for voice_id in temp.values():      # always remove the temporary clones
            try:
                requests.delete(f"{API}/voices/{voice_id}", headers={"xi-api-key": key}, timeout=60)
            except requests.RequestException:
                print("Could not delete temporary voice", voice_id, flush=True)
    s["generations"] += 1
    save_state(s)
    return jsonify(ok=True)


# ---- the test ------------------------------------------------------------
@app.get("/clip/<int:i>/<int:slot>")
def clip(i, slot):
    """Serve clip A, B, C or D (slot 0-3) for sentence i without revealing which is real."""
    order = load_state()["order"].get(str(i))
    if not order or slot not in range(len(order)):
        abort(404)
    kind = order[slot]
    return send_file(clip_path(kind, i), mimetype="audio/wav" if kind == "real" else "audio/mpeg")


@app.get("/api/answer/<int:i>")
def answer(i):
    order = load_state()["order"].get(str(i))
    if not order:
        abort(404)
    return jsonify(order=order)


# ---- clean up ------------------------------------------------------------
@app.post("/api/delete-clone")
def delete_clone():
    """Delete the clone from ElevenLabs and all generated clips. Recordings are kept."""
    s = load_state()
    if s["voice_id"]:
        resp = requests.delete(f"{API}/voices/{s['voice_id']}", headers={"xi-api-key": api_key()}, timeout=60)
        if resp.status_code != 404:      # 404 means it was already gone
            check(resp)
    for f in g.dir.glob("*_*.mp3"):
        f.unlink()
    s.update(voice_id=None, order={}, generations=0)
    save_state(s)
    return jsonify(ok=True)


@app.post("/api/delete-recordings")
def delete_recordings():
    """Delete your recorded voice. The clone and generated clips are kept."""
    for f in g.dir.glob("real_*.wav"):
        f.unlink()
    s = load_state()
    s["order"] = {}     # the test needs the real clips, so it has to be set up again
    save_state(s)
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(debug=False)
