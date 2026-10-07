"""Remove visitors who have been away for a while, including their ElevenLabs clone.

    python cleanup.py 7      (visitors untouched for 7 days; the default)

Run it now and then, or from a daily scheduled job, so abandoned clones
don't use up your ElevenLabs voice slots.
"""
import json
import os
import shutil
import sys
import time

import requests

from app import API, DATA

days = float(sys.argv[1]) if len(sys.argv) > 1 else 7
cutoff = time.time() - days * 86400
key = os.environ["ELEVENLABS_API_KEY"]

for folder in DATA.glob("*"):
    if not folder.is_dir():
        continue
    if max((f.stat().st_mtime for f in folder.iterdir()), default=0) > cutoff:
        continue
    state = folder / "state.json"
    voice = json.loads(state.read_text()).get("voice_id") if state.exists() else None
    if voice:
        r = requests.delete(f"{API}/voices/{voice}", headers={"xi-api-key": key}, timeout=60)
        if not r.ok and r.status_code != 404:
            print(f"kept {folder.name}: ElevenLabs said {r.status_code}")
            continue
    shutil.rmtree(folder)
    print("removed", folder.name)
