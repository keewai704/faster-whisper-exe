# faster-whisper-exe

Whisper **large-v3** (CTranslate2 int8) を 1 つの `faster-whisper.exe` にまとめた Windows 用 CLI です。
NVIDIA GPU があれば CUDA で、なければ CPU で動きます。

## 使い方

```
faster-whisper.exe input.mp4 -l ja -f srt
faster-whisper.exe a.wav b.mp3 -f all -o out --vad_filter
faster-whisper.exe --info
```

| オプション | 説明 |
|---|---|
| `-l, --language` | 言語コード (`ja`, `en` …)。省略時は自動判定 |
| `--task` | `transcribe` / `translate` (英訳) |
| `-f, --output_format` | `txt` `srt` `vtt` `json` `all` `none` (既定 `srt`) |
| `-o, --output_dir` | 出力先 (既定は入力ファイルと同じ場所) |
| `--device` | `auto` `cuda` `cpu` |
| `--compute_type` | 既定は CUDA で `float16`、CPU で `int8` |
| `--vad_filter` | 無音区間をスキップ |
| `--word_timestamps` | 単語単位のタイムスタンプ |

初回起動時に中身を `%LOCALAPPDATA%\faster-whisper-exe\<ビルドID>` へ展開します。2 回目以降は展開済みのものを使うので、すぐ起動します。

## CUDA

必要なのは NVIDIA ドライバだけです。CUDA Toolkit や cuDNN のインストールは要りません。exe には cuBLAS (`cublas64_12.dll`, `cublasLt64_12.dll`) が入っています。
ctranslate2 は Whisper の推論で cuDNN を読み込まないため、cuDNN (約 1.1 GB) は同梱していません。

## 量子化と Blackwell (RTX 50 系)

モデルは int8 の重みで保存しています (exe を小さくするため)。CUDA ではロード時に float16 へ展開して計算します。
ctranslate2 は FP8 / NVFP4 に対応していないため、Blackwell で選べるのは int8・fp16・bf16 系だけです。
RTX 5070 Ti で large-v3 に 132 秒の音声を与え、beam 5 で測った結果は次のとおりです。

| compute_type | 時間 | 備考 |
|---|---|---|
| `float16` (既定) | 3.98 s | |
| `int8_float16` | 4.80 s | Blackwell では int8 GEMM のほうが遅い |
| `bfloat16` | 2.91 s | 速いが、一部の区間が丸ごと欠落した |
| `int8_bfloat16` | 3.46 s | 同上 |

## ビルド

GitHub Actions (`.github/workflows/build.yml`) が次の順に処理します。

1. `openai/whisper-large-v3` を CTranslate2 の `int8_float16` に変換する (結果はキャッシュ)
2. Nuitka で onefile ビルドする (zstd レベル 22 で圧縮)
3. CPU でスモークテストを実行し、exe を artifact としてアップロードする

ローカルでビルドする場合:

```
pip install -r requirements.txt
python convert_model.py --output_dir model   # torch / transformers が必要
python build.py --model_dir model --compiler=--msvc=latest
```
