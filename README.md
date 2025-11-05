# Tsundere AI Chatbot

A lightweight bilingual chatbot fine-tuned from **Qwen2.5-1.5B-Instruct** using LoRA.  
It runs locally with **Gradio UI**, supports both Chinese and English, and simulates a playful “tsundere” personality.

---

## Features
- Fine-tuned with LoRA personality
- Bilingual chat (Chinese / English)
- Local run — no API required
- Simple Gradio web interface
- Easy retraining with your own data (`data/train.json`)

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the chatbot
python src/chat_app.py
```

Then open your browser and visit:
```
http://127.0.0.1:7860
```

---

## Fine-tuning (Optional)

To retrain or adapt with new dialogue data:
```bash
python src/train_lora.py
```

The fine-tuned model will be saved to:
```
./chat_model_lora/
```

---

## Project Structure
```
chat_model_lora/   # LoRA fine-tuned weights
data/train.json    # Training dataset
src/               # Training scripts + chat interface
requirements.txt   # Dependencies
README.md          # Project description
```

---

## Author
Developed by **Hushmonday**  
For educational and personal use only.
