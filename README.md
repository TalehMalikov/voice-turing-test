# Voice Turing test

Can you tell your real voice from a cloned one?

Record a few sentences, and the app clones your voice with ElevenLabs. Then
you hear four clips per sentence, shuffled: your real voice, your clone, and
two stock AI voices. Guess which one is you.

**Try it:** [turing.tmalikov.com](https://turing.tmalikov.com)

## Run it locally

You'll need your own [ElevenLabs](https://elevenlabs.io) API key, on a plan
that includes Instant Voice Cloning.

```
pip install -r requirements.txt

copy .env.example .env     
# paste your key into .env

python app.py
```

Then open http://localhost:5000.

## Privacy

Your recordings and clone are saved on the server for your browser only.
Delete them any time with the buttons at the bottom of the page.
