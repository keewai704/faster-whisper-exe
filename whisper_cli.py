"""faster-whisper large-v3 (int8) single-binary CLI."""

import argparse
import json
import os
import sys
import time


def _base_dir():
    if getattr(sys, "frozen", False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


def _setup_dll_paths():
    # Make the bundled cuBLAS / cuDNN DLLs visible to ctranslate2.
    base = _base_dir()
    for d in (os.path.join(base, "cuda"), base):
        if os.path.isdir(d):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(d)


_setup_dll_paths()

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import ctranslate2  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402


def fmt_ts(seconds, sep):
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def write_output(path, fmt, segments, info):
    with open(path, "w", encoding="utf-8") as f:
        if fmt == "txt":
            for seg in segments:
                f.write(seg.text.strip() + "\n")
        elif fmt == "srt":
            for i, seg in enumerate(segments, 1):
                f.write(f"{i}\n{fmt_ts(seg.start, ',')} --> {fmt_ts(seg.end, ',')}\n{seg.text.strip()}\n\n")
        elif fmt == "vtt":
            f.write("WEBVTT\n\n")
            for seg in segments:
                f.write(f"{fmt_ts(seg.start, '.')} --> {fmt_ts(seg.end, '.')}\n{seg.text.strip()}\n\n")
        elif fmt == "json":
            json.dump(
                {
                    "language": info.language,
                    "language_probability": info.language_probability,
                    "duration": info.duration,
                    "segments": [
                        {"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments
                    ],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


def load_model(args):
    model_dir = args.model_dir or os.path.join(_base_dir(), "model")
    device = args.device
    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"

    compute_type = args.compute_type
    if compute_type == "auto":
        # The int8 weights are expanded to float16 on load: on Blackwell (RTX 50)
        # fp16 GEMMs beat int8_float16 (~20% faster on an RTX 5070 Ti). bfloat16
        # is faster still but drops segments, so it is opt-in only.
        compute_type = "float16" if device == "cuda" else "int8"

    try:
        model = WhisperModel(model_dir, device=device, compute_type=compute_type, cpu_threads=args.threads)
    except Exception as e:
        if device != "cuda" or args.device == "cuda":
            raise
        print(f"[warn] CUDA init failed ({e}); falling back to CPU", file=sys.stderr)
        device, compute_type = "cpu", "int8"
        model = WhisperModel(model_dir, device=device, compute_type=compute_type, cpu_threads=args.threads)

    print(f"[info] device={device} compute_type={compute_type}", file=sys.stderr)
    return model


def main():
    p = argparse.ArgumentParser(
        prog="faster-whisper",
        description="Transcribe audio/video with Whisper large-v3 (int8, bundled).",
    )
    p.add_argument("audio", nargs="*", help="input audio/video files")
    p.add_argument("-l", "--language", default=None, help="language code, e.g. ja, en (default: auto-detect)")
    p.add_argument("--task", choices=["transcribe", "translate"], default="transcribe")
    p.add_argument("-f", "--output_format", choices=["txt", "srt", "vtt", "json", "all", "none"], default="srt")
    p.add_argument("-o", "--output_dir", default=None, help="output directory (default: next to input)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument(
        "--compute_type",
        default="auto",
        help="float16, int8_float16, bfloat16, int8, float32 ... (default: float16 on CUDA, int8 on CPU)",
    )
    p.add_argument("--beam_size", type=int, default=5)
    # Defaults measured on a 24 min Japanese lecture: VAD + no context carry-over
    # removed Whisper's repeated sentences and the hallucinated outro, and was the
    # fastest and closest to MAI-Transcribe 2 of the settings tried.
    p.add_argument(
        "--vad_filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip silence with Silero VAD (default: on; --no-vad_filter to disable)",
    )
    p.add_argument(
        "--condition_on_previous_text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="feed the previous text as context (default: off, which avoids repetition loops)",
    )
    p.add_argument("--initial_prompt", default=None)
    p.add_argument("--word_timestamps", action="store_true")
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0 = default)")
    p.add_argument("--model_dir", default=None, help="use another CTranslate2 Whisper model directory")
    p.add_argument("--info", action="store_true", help="print runtime info and exit")
    args = p.parse_args()

    if args.info:
        print(f"ctranslate2 {ctranslate2.__version__}")
        print(f"cuda devices: {ctranslate2.get_cuda_device_count()}")
        print(f"cuda compute types: {ctranslate2.get_supported_compute_types('cuda') if ctranslate2.get_cuda_device_count() else '-'}")
        print(f"model dir: {args.model_dir or os.path.join(_base_dir(), 'model')}")
        return 0

    if not args.audio:
        p.print_help()
        return 1

    model = load_model(args)
    formats = ["txt", "srt", "vtt", "json"] if args.output_format == "all" else [args.output_format]

    for audio in args.audio:
        if not os.path.isfile(audio):
            print(f"[error] not found: {audio}", file=sys.stderr)
            continue
        t0 = time.time()
        segments_iter, info = model.transcribe(
            audio,
            language=args.language,
            task=args.task,
            beam_size=args.beam_size,
            vad_filter=args.vad_filter,
            condition_on_previous_text=args.condition_on_previous_text,
            initial_prompt=args.initial_prompt,
            word_timestamps=args.word_timestamps,
        )
        print(f"[info] {audio}: language={info.language} ({info.language_probability:.2f}) duration={info.duration:.1f}s", file=sys.stderr)
        segments = []
        for seg in segments_iter:
            segments.append(seg)
            print(f"[{fmt_ts(seg.start, '.')} --> {fmt_ts(seg.end, '.')}] {seg.text.strip()}", flush=True)

        if "none" not in formats:
            out_dir = args.output_dir or os.path.dirname(os.path.abspath(audio))
            os.makedirs(out_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(audio))[0]
            for fmt in formats:
                out = os.path.join(out_dir, f"{stem}.{fmt}")
                write_output(out, fmt, segments, info)
                print(f"[info] wrote {out}", file=sys.stderr)
        print(f"[info] done in {time.time() - t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
