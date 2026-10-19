"""Compare faster-whisper.exe (local GPU) with an OpenRouter STT model.

The OpenRouter request is pinned to zero-data-retention endpoints
(provider.zdr = true, provider.data_collection = "deny"), so a provider that
would keep the audio is never used: the request fails instead.

Audio is sent as 16 kHz mono FLAC in chunks (the Azure provider rejects m4a).

The API key is read from OPENROUTER_API_KEY or %USERPROFILE%\\.openrouter_key
and is never printed.

Without a reference transcript the script reports speed and the character
error rate of each system against the other (agreement), plus both texts.
Pass --reference to score both against a human transcript.
"""

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request

import av
import numpy as np

API = "https://openrouter.ai/api/v1/audio/transcriptions"


def api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        path = os.path.join(os.path.expanduser("~"), ".openrouter_key")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                key = f.read().strip()
    if key and "=" in key.splitlines()[0]:  # KEY=value .env line
        key = key.splitlines()[0].split("=", 1)[1].strip().strip("\"'")
    if not key:
        sys.exit("set OPENROUTER_API_KEY or put the key in ~/.openrouter_key")
    return key


def decode_16k(path):
    """Decode any audio to 16 kHz mono int16 PCM."""
    with av.open(path) as src:
        stream = src.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        pcm = []
        for frame in src.decode(stream):
            for f in resampler.resample(frame):
                pcm.append(f.to_ndarray().reshape(-1))
        for f in resampler.resample(None):
            pcm.append(f.to_ndarray().reshape(-1))
    return np.concatenate(pcm).astype(np.int16)


def to_flac(pcm):
    buf = io.BytesIO()
    with av.open(buf, "w", format="flac") as dst:
        st = dst.add_stream("flac", rate=16000, layout="mono")
        frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
        frame.rate = 16000
        for packet in st.encode(frame):
            dst.mux(packet)
        for packet in st.encode(None):
            dst.mux(packet)
    return buf.getvalue()


def run_openrouter(model, pcm, language, chunk_s):
    """Send 16 kHz FLAC chunks one by one (m4a is rejected by the Azure provider)."""
    key = api_key()
    texts, cost, wall = [], 0.0, 0.0
    step = chunk_s * 16000
    for start in range(0, len(pcm), step):
        body = {
            "model": model,
            "input_audio": {"data": base64.b64encode(to_flac(pcm[start:start + step])).decode(), "format": "flac"},
            "provider": {"zdr": True, "data_collection": "deny"},
        }
        if language:
            body["language"] = language
        req = urllib.request.Request(
            API,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                res = json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"OpenRouter HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")
        wall += time.time() - t
        texts.append(res.get("text", ""))
        cost += (res.get("usage") or {}).get("cost", 0) or 0
        print(f"  chunk {start // step + 1}: {time.time() - t:.1f}s", flush=True)
    return " ".join(texts), wall, {"cost_usd": cost, "chunks": len(texts)}


def run_local(exe, audio, language, extra):
    out_dir = tempfile.mkdtemp()
    cmd = [exe, audio, "-f", "json", "-o", out_dir, *extra]
    if language:
        cmd += ["-l", language]
    t = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.time() - t
    if proc.returncode:
        sys.exit(f"local exe failed:\n{proc.stderr[-2000:]}")
    stem = os.path.splitext(os.path.basename(audio))[0]
    with open(os.path.join(out_dir, stem + ".json"), encoding="utf-8") as f:
        data = json.load(f)
    done = [l for l in proc.stderr.splitlines() if "done in" in l]
    infer = float(done[-1].split("done in")[1].strip().rstrip("s")) if done else None
    return " ".join(s["text"] for s in data["segments"]), wall, infer


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in text if unicodedata.category(c)[0] in "LN")


def cer(ref, hyp):
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return float("nan")
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--exe", default="faster-whisper.exe")
    p.add_argument("--model", default="microsoft/mai-transcribe-2")
    p.add_argument("-l", "--language", default=None)
    p.add_argument("--reference", default=None, help="text file with the correct transcript")
    p.add_argument("--report", default=None, help="write a JSON report here")
    p.add_argument("--chunk_seconds", type=int, default=600, help="audio length per OpenRouter request")
    p.add_argument("--local_args", nargs=argparse.REMAINDER, default=[], help="extra args for the exe")
    args = p.parse_args()

    pcm = decode_16k(args.audio)
    duration = len(pcm) / 16000
    print(f"audio: {duration:.1f}s", flush=True)

    local_text, local_wall, local_infer = run_local(args.exe, args.audio, args.language, args.local_args)
    print(f"local : wall {local_wall:.1f}s, transcribe {local_infer}s", flush=True)
    or_text, or_wall, usage = run_openrouter(args.model, pcm, args.language, args.chunk_seconds)
    print(f"{args.model}: {or_wall:.1f}s  usage={usage}", flush=True)

    report = {
        "audio_seconds": duration,
        "local": {"wall_s": local_wall, "transcribe_s": local_infer, "text": local_text},
        "openrouter": {"model": args.model, "wall_s": or_wall, "usage": usage, "text": or_text},
        "cer_local_vs_openrouter": cer(or_text, local_text),
    }
    if args.reference:
        with open(args.reference, encoding="utf-8") as f:
            ref = f.read()
        report["cer_local"] = cer(ref, local_text)
        report["cer_openrouter"] = cer(ref, or_text)

    for k in ("cer_local_vs_openrouter", "cer_local", "cer_openrouter"):
        if k in report:
            print(f"{k}: {report[k]:.2%}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"report: {args.report}")


if __name__ == "__main__":
    main()
