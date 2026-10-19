# faster-whisper-exe

Whisper **large-v3** (CTranslate2 int8) を 1 つの `faster-whisper.exe` にまとめた Windows 用 CLI です。
NVIDIA GPU があれば CUDA で、なければ CPU で動きます。

## 使い方

```
faster-whisper.exe input.mp4 -l ja -f srt
faster-whisper.exe a.wav b.mp3 -f all -o out
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
| `--no-vad_filter` | 無音区間のスキップ (VAD) を止める。既定はオン |
| `--condition_on_previous_text` | 直前の文を文脈として渡す。既定はオフ (同じ文を繰り返す不具合を防ぐため) |
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

## MAI-Transcribe 2 との比較

24 分 13 秒の日本語の講義録音 (Notability の録音) で、`bench/compare_openrouter.py` を使って比較しました。
OpenRouter には `provider.zdr = true` と `data_collection = "deny"` を付けて送っています。データを保持しないエンドポイントが使えない場合は、送信せずにエラーで止まります。

| | 処理時間 | MAI との差 (CER) | 同じ文の繰り返し |
|---|---|---|---|
| faster-whisper.exe (RTX 5070 Ti、旧既定: VAD なし・文脈あり) | 80 s | 18.9 % | 5 回 + 存在しない締めの一文 |
| faster-whisper.exe (新既定: VAD あり・文脈なし) | 59 s | 16.4 % | なし |
| `microsoft/mai-transcribe-2` (OpenRouter) | 18 s | — | なし (費用 $0.04) |

正解の文字起こしがないので、CER は MAI との食い違いを表します。MAI は「えー」「まあ」などのつなぎ言葉も残すため、その分だけ数字が大きくなります。
固有名詞や専門用語 (那覇空港、全日空機、捜索救難など) は MAI のほうが正確でした。
hotwords で専門用語を指定すると、Whisper は繰り返しに陥りました (CER 58 %) ので使っていません。

```
python bench/extract_pdf_audio.py note.pdf -o audio      # PDF に埋め込まれた音声を取り出す
python bench/compare_openrouter.py audio.m4a --exe faster-whisper.exe -l ja --report report.json
```

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
