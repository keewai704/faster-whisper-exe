"""Extract audio recordings embedded in a PDF (e.g. a Notability export with audio).

Looks at both document-level embedded files and file-attachment / rich-media
annotations, and writes every audio-looking stream to the output directory.
"""

import argparse
import os

from pypdf import PdfReader

AUDIO_EXT = (".m4a", ".aac", ".mp3", ".wav", ".caf", ".mp4", ".ogg", ".opus", ".flac", ".aiff", ".3gp")
MAGIC = {b"ftyp": ".m4a", b"ID3": ".mp3", b"RIFF": ".wav", b"caff": ".caf", b"OggS": ".ogg", b"fLaC": ".flac"}


def guess_ext(name, data):
    ext = os.path.splitext(name or "")[1].lower()
    if ext in AUDIO_EXT:
        return ext
    if data[4:8] == b"ftyp":
        return ".m4a"
    for magic, e in MAGIC.items():
        if data.startswith(magic):
            return e
    return None


def iter_embedded(reader):
    for name, contents in (reader.attachments or {}).items():
        for i, data in enumerate(contents):
            yield f"{name}" if i == 0 else f"{i}_{name}", data
    for pno, page in enumerate(reader.pages):
        for annot in page.get("/Annots") or []:
            a = annot.get_object()
            spec = a.get("/FS")
            if spec is None and a.get("/RichMediaContent"):
                assets = a["/RichMediaContent"].get_object().get("/Assets")
                names = assets.get_object().get("/Names", []) if assets else []
                for j in range(0, len(names), 2):
                    yield f"p{pno + 1}_{names[j]}", names[j + 1].get_object()["/EF"]["/F"].get_object().get_data()
                continue
            if spec is None:
                continue
            spec = spec.get_object()
            ef = spec.get("/EF")
            if not ef:
                continue
            name = spec.get("/UF") or spec.get("/F") or "attachment"
            yield f"p{pno + 1}_{name}", ef["/F"].get_object().get_data()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pdf")
    p.add_argument("-o", "--output_dir", default=".")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    reader = PdfReader(args.pdf)
    found = 0
    seen = set()
    for name, data in iter_embedded(reader):
        ext = guess_ext(name, data)
        if not ext or hash(data) in seen:
            continue
        seen.add(hash(data))
        base = os.path.splitext(os.path.basename(str(name)))[0] or "audio"
        out = os.path.join(args.output_dir, f"{found:02d}_{base}{ext}")
        with open(out, "wb") as f:
            f.write(data)
        print(f"{len(data) / 1e6:8.2f} MB  {out}")
        found += 1
    if not found:
        raise SystemExit("no embedded audio found")


if __name__ == "__main__":
    main()
