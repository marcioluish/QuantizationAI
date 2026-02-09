# Meeting Minutes Generator

Generate professional meeting minutes from audio recordings using multiple AI models, compare their outputs, and get AI-powered analysis—all in one Gradio app. Supports configurable quantization so you can run larger models on limited GPU memory.

---

## What This App Does

1. **Transcribes** your meeting audio (speech-to-text).
2. **Generates minutes** with one or more open-weight LLMs (Llama, Mistral, Qwen, DeepSeek, etc.) using optional 4-bit/8-bit quantization.
3. **Compares and analyzes** the generated minutes with an AI judge (GPT-4o-mini).

You can run several models on the same transcript and see performance and memory stats, then use the Analysis tab to get a comparison or feedback.

---

## AI Models Used (and for what)

| Role | Model / API | Purpose |
|------|-------------|---------|
| **Transcription** | **OpenAI Whisper** (`openai/whisper-medium.en`) | Converts your uploaded audio file into text. Runs locally via Hugging Face Transformers (no OpenAI API for transcription). |
| **Minute generation** | **Open-weight LLMs** (see list below) | Turn the transcript into structured meeting minutes (summary, discussion points, takeaways, action items). Models run locally on your GPU with optional 4-bit/8-bit quantization. |
| **Analysis / comparison** | **OpenAI GPT-4o-mini** (API) | Compares multiple minutes and ranks them, or gives feedback on a single set of minutes. Requires an OpenAI API key. |

### Minute-generation models (pick one or more)

- **Meta:** Llama 3.1 8B Instruct, Llama 3.2 3B Instruct  
- **Mistral AI:** Mistral 7B Instruct v0.3  
- **Qwen:** Qwen 2.5 3B Instruct, Qwen 2.5 7B Instruct  
- **DeepSeek:** DeepSeek R1 Distill (Llama 8B, Qwen 1.5B, Qwen 7B)

Some of these are gated on the Hugging Face Hub and require prior approval and a Hugging Face token.

---

## Keys and Secrets You Need

To run the project you need **at least one** of the following, depending on what you use:

| Key / secret | Required for | Where to get it |
|--------------|--------------|------------------|
| **`HF_TOKEN`** | Downloading gated models (e.g. Llama, Mistral) and private repos | [Hugging Face → Settings → Access Tokens](https://huggingface.co/settings/tokens). Create a token with “Read” access. |
| **`OPENAI_API_KEY`** | Analysis tab (GPT-4o-mini comparison and feedback) | [OpenAI → API keys](https://platform.openai.com/api-keys). Billing must be enabled for API use. |

- **Transcription (Whisper)** and **minute generation (LLMs)** work with only `HF_TOKEN` (for gated models).  
- The **Analysis** tab only works if `OPENAI_API_KEY` is set.

### How to provide the keys

**Google Colab (recommended)**  
- Use **Secrets**: left sidebar → 🔑 Secrets.  
- Add:  
  - `HF_TOKEN` = your Hugging Face token  
  - `OPENAI_API_KEY` = your OpenAI API key  

**Local / other environments**  
- Set environment variables before starting the app:  
  - `HF_TOKEN`  
  - `OPENAI_API_KEY`  

Example (Linux/macOS):

```bash
export HF_TOKEN="your-hf-token"
export OPENAI_API_KEY="your-openai-key"
```

---

## Requirements

- **Python 3.10+**
- **GPU** (NVIDIA with CUDA; e.g. T4 on Colab). CPU-only is possible but slow for LLMs.
- **ffmpeg** (for audio handling; often preinstalled on Colab)

---

## Setup and run

### 1. Clone and install

```bash
git clone https://github.com/marcioluish/QuantizationAI.git
cd QuantizationAI
pip install -r requirements.txt
```

### 2. Set keys (see above)

Colab: add `HF_TOKEN` and `OPENAI_API_KEY` to Secrets.  
Local: export `HF_TOKEN` and `OPENAI_API_KEY`.

### 3. Launch the app

**From the notebook (e.g. in Colab):**

Open `main.ipynb`, run the setup cells (clone, install, GPU check), then run the cell that does:

```python
from app import launch_app
launch_app()
```

**From the command line:**

```bash
python app.py
```

The app will open in the browser (and, when run from Colab, you’ll get a shareable link).

---

## Gated models (Llama, Mistral, etc.)

Some Hugging Face models require you to accept their terms on the Hub before use:

- [Meta Llama](https://huggingface.co/meta-llama)  
- [Mistral](https://huggingface.co/mistralai)  

Log in on the Hub, open the model page, and accept the license. Use the same account for `HF_TOKEN`.

---

## Project layout

Code is split into two packages for a cleaner structure:

| Location | Contents |
|----------|----------|
| **Root** | `app.py` (entry point: builds and launches the Gradio app), `main.ipynb`, `requirements.txt`, `README.md`. |
| **`ui/`** | Gradio UI layer: `app_state.py`, `ui_helpers.py`, `ui_tabs.py`, `handlers_config.py`, `handlers_generate.py` (state, layout, event handlers). |
| **`engine/`** | Business logic: `transcription.py` (Whisper), `minute_generation.py` (LLMs), `analysis.py` (GPT-4o-mini), `stats.py`, `quantization.py`, `model_configs.py`, `utils.py`. |
| **`docs/`** | `FOLDER_ORGANIZATION.md` — notes on the folder structure. |

The root `app.py` imports from `ui` and `engine`; the notebook and CLI still use `from app import launch_app` and `python app.py` respectively.

---

## License and attribution

- **Whisper:** OpenAI; used under the model’s license (see [Hugging Face model card](https://huggingface.co/openai/whisper-medium.en)).  
- **LLMs:** See each model’s card on the Hugging Face Hub (Meta, Mistral, Qwen, DeepSeek, etc.).  
- **GPT-4o-mini:** Subject to [OpenAI’s terms and usage policies](https://openai.com/policies).

This project is for demonstration and educational use. Ensure your use of the above models and APIs complies with their respective terms.
