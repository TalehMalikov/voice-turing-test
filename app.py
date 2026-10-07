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
import wave
from pathlib import Path

import numpy as np
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
    s = {"sentences": DEFAULT_SENTENCES, "extra": DEFAULT_EXTRA, "voice_id": None, "order": {}, "generations": 0, "pitches": {}}
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


FEMALE_PITCH_HZ = 165   # typical speaking pitch: men sit below this, women above


def median_pitch(path):
    """Median pitch in Hz of the voiced parts of a 16-bit WAV file, or 0 if there isn't a clear one."""
    with wave.open(str(path)) as w:
        sr, channels = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    x = x[::channels]
    step = max(1, round(sr / 8000))                  # work at about 8 kHz, it's plenty
    rate = sr / step
    d = x[: len(x) // step * step].reshape(-1, step).mean(axis=1)
    win, hop = int(rate * 0.04), int(rate * 0.02)
    min_lag, max_lag = int(rate / 400), int(rate / 70)
    count = max_lag + 2 - min_lag
    found = []
    for s in range(0, len(d) - win - max_lag - 2, hop):
        a = d[s:s + win]
        e0 = float(a @ a)
        if e0 / win < 1e-4:                          # too quiet: a pause or a breath
            continue
        windows = np.lib.stride_tricks.sliding_window_view(d[s + min_lag: s + min_lag + count + win - 1], win)
        r = (windows @ a) / np.sqrt(e0 * (windows ** 2).sum(axis=1) + 1e-9)
        best = r[:-1].max()
        if best < 0.6:                               # not clearly a voiced sound
            continue
        for k in range(1, count - 1):                # first strong peak avoids octave errors
            if r[k] >= 0.9 * best and r[k] >= r[k - 1] and r[k] >= r[k + 1]:
                found.append(rate / (min_lag + k))
                break
    return float(np.median(found)) if len(found) >= 5 else 0.0


def detect_gender(s):
    """Guess male or female from the median pitch across the visitor's recordings."""
    pitches = sorted(s["pitches"].values())
    if not pitches:
        return None
    median = pitches[len(pitches) // 2]
    return {"gender": "female" if median >= FEMALE_PITCH_HZ else "male", "pitch": round(median)}


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
        detected=detect_gender(s),
        has_voice=bool(s["voice_id"]),
        has_key=bool(os.environ.get("ELEVENLABS_API_KEY")),
    )


@app.post("/api/sentences")
def set_sentences():
    clean = lambda items: [x.strip() for x in items if x.strip()]
    s = load_state()
    sentences = clean(request.json["sentences"])
    # If the page didn't send the new sentences (an older cached page), keep the ones we have.
    extra = clean(request.json["extra"]) if "extra" in request.json else s["extra"]
    if not sentences:
        abort(400, "Need at least one sentence to record.")
    if len(sentences) > MAX_SENTENCES:
        abort(400, f"Record at most {MAX_SENTENCES} sentences.")
    if len(extra) > MAX_EXTRA:
        abort(400, f"Use at most {MAX_EXTRA} new sentences.")
    if any(len(x) > MAX_SENTENCE_CHARS for x in sentences + extra):
        abort(400, f"Keep each sentence under {MAX_SENTENCE_CHARS} characters.")
    if sentences != s["sentences"]:                  # recordings only match the old text
        for f in g.dir.glob("real_*.wav"):
            f.unlink()
        s["pitches"] = {}
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
    try:
        pitch = median_pitch(real_path(i))
    except (wave.Error, EOFError, ValueError):       # not a readable recording: skip the guess
        pitch = 0
    if pitch:
        s["pitches"][str(i)] = round(pitch, 1)
    else:
        s["pitches"].pop(str(i), None)
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
    s["pitches"] = {}
    s["order"] = {}     # the test needs the real clips, so it has to be set up again
    save_state(s)
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(debug=False)
