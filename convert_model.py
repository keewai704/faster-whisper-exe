"""Convert openai/whisper-large-v3 to a CTranslate2 model with int8 weights.

int8_float16 keeps the non-quantized tensors (layer norms, biases) in float16,
which is smaller on disk than plain int8 (float32 leftovers). The model still
loads on CPU as int8 and on CUDA as int8_float16.
"""

import argparse
import os

from ctranslate2.converters import TransformersConverter

p = argparse.ArgumentParser()
p.add_argument("--model", default="openai/whisper-large-v3")
p.add_argument("--output_dir", default="model")
p.add_argument("--quantization", default="int8_float16")
args = p.parse_args()

TransformersConverter(
    args.model,
    copy_files=["tokenizer.json", "preprocessor_config.json"],
    low_cpu_mem_usage=True,
).convert(args.output_dir, quantization=args.quantization, force=True)

for name in sorted(os.listdir(args.output_dir)):
    size = os.path.getsize(os.path.join(args.output_dir, name))
    print(f"{size / 1e6:10.1f} MB  {name}")
