# STEP 3: Get llama.cpp, then quantize

llama.cpp provides the tools that do the real conversion and compression.
You need **two things** from it:

| Need | For |
|---|---|
| `convert_hf_to_gguf.py` (a script) | raw model -> F16 GGUF |
| `llama-quantize` and `llama-perplexity` (programs) | compress the model; measure quality |

## 3A. Get the script (needs only git)

Clone it **inside this project folder** so the scripts find it automatically:

```
git clone --depth 1 https://github.com/ggml-org/llama.cpp
pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
```

(The second line installs `torch`, `transformers`, `gguf`, ... and is a large download. It is only needed for the one-off conversion.)

## 3B. Get the programs, pick ONE option

### Option 1: prebuilt release (no CMake, no compiler). Recommended.
1. Open https://github.com/ggml-org/llama.cpp/releases
2. Download the latest Windows CPU zip (name like `llama-bXXXX-bin-win-cpu-x64.zip`) and unzip it, for example to `C:\llama-bin`.
3. Tell the scripts where it is (PowerShell):
   ```
   $env:LLAMA_BIN_DIR = "C:\llama-bin"
   ```
   Check it contains `llama-quantize.exe` and `llama-perplexity.exe`.

### Option 2: build it yourself (needs CMake + Visual Studio Build Tools)
```
cd llama.cpp
cmake -B build
cmake --build build --config Release
cd ..
```
The scripts look in `llama.cpp\build\bin\Release` automatically.

## 3C. Run the sweep

```
python step3_quantize_sweep.py
```

This will:
1. convert `models/tinyllama-raw` -> `models/tinyllama-f16.gguf`
2. quantize it to Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M and Q2_K
3. print a size table

Handy options: `--levels Q4_K_M` (only one level), `--force` (rebuild).

Expected files in `models/`:

```
tinyllama-raw/        original safetensors (~2.2 GB)
tinyllama-f16.gguf    16-bit reference (~2.2 GB)
tinyllama-q80.gguf    tinyllama-q6k.gguf    tinyllama-q5km.gguf
tinyllama-q4km.gguf   tinyllama-q3km.gguf   tinyllama-q2k.gguf
```

Disk needed: roughly 6 GB in total.

## Troubleshooting
- `Could not find 'llama-quantize'`: set `LLAMA_BIN_DIR` (Option 1) or build it (Option 2).
- `convert_hf_to_gguf.py not found`: the `llama.cpp` folder must be inside this project, or set `LLAMA_CPP_DIR`.
- Conversion import errors: re-run the `pip install -r ...requirements-convert_hf_to_gguf.txt` line in the activated venv.

Next: `python step4_benchmark.py`
