Tsundere AI Chatbot

A local bilingual AI assistant that can both flirt and code.
Built with Qwen2.5 and LoRA fine-tuning — supports personalized chat and Python code execution.

Overview

Tsundere AI Chatbot is a locally deployed intelligent chat system that combines
LLM fine-tuning, LoRA parameter-efficient training, and real-time Python code execution.

It allows users to chat naturally in both Chinese and English,
while also generating and running executable Python code —
making it a blend of a charming AI companion and a practical coding assistant.

✨ Key Features
Feature	Description
Tsundere Personality Chat	Custom fine-tuned conversational style with a bilingual “tsundere” (playfully sassy) tone.
LoRA Fine-Tuning	Train and personalize the model efficiently using lightweight Low-Rank Adaptation.
Model Merging & Loading	Automatically detects and merges LoRA weights for faster, memory-efficient inference.
Python Code Execution	Recognizes and executes code blocks in real-time (e.g., python ... ).
Multi-Turn Context	Maintains full conversation history for coherent and natural dialogue.
Fully Local Deployment	Runs 100% offline — no API calls, ensuring privacy and security.
Tech Stack
Category	Tools / Frameworks
Model Framework	Hugging Face Transformers

Fine-Tuning Method	PEFT / LoRA

Data Format	JSON (role-based conversation data)
Base Model	Qwen2.5-1.5B-Instruct

UI Framework	Gradio ChatInterface

Language	Python 3.10+
Environment	Compatible with both CPU and GPU
Project Structure
chatbot/
│
├── data/
│   └── train.json                # Training dataset (custom bilingual dialogue)
│
├── checkpoints_lora/             # Intermediate LoRA checkpoints
├── chat_model_lora/              # Fine-tuned LoRA model
├── chat_model/                   # Base model (Qwen2.5)
│
├── src/
│   ├── train.py                  # Base model training script
│   ├── train_lora.py             # LoRA fine-tuning script
│   ├── chat_app.py               # Chat interface (conversation + code execution)
│   └── utils.py                  # Utility functions
│
├── requirements.txt              # Dependencies
└── README.md                     # Documentation

Installation
1️⃣ Install dependencies
git clone https://github.com/yourusername/tsundere-chatbot.git
cd tsundere-chatbot
pip install -r requirements.txt


💡 On Windows, the script automatically disables bitsandbytes to avoid CUDA DLL issues.

2️⃣ Prepare the model

Download and place your models under:

chat_model/           # Qwen2.5-1.5B-Instruct base model
chat_model_lora/      # LoRA fine-tuned model (optional)


If you only have the base model, fine-tune it yourself using:

python src/train_lora.py

3️⃣ Launch the chatbot
python src/chat_app.py


Once started, you’ll see something like:

* Running on local URL: http://127.0.0.1:7860


Then open http://127.0.0.1:7860
 in your browser.

💬 Chat Examples
1️⃣ Regular Chat
User: 你好  
Bot: Hmph! It’s not like I missed you or anything… hello! 😤

2️⃣ Python Code Generation
User: Write a Python function to print the first 10 Fibonacci numbers.
Bot:
```python
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        print(a)
        a, b = b, a + b

fibonacci(10)


🧠 Execution Result:

0  
1  
1  
2  
3  
5  
8  
13  
21  
34

🔬 Training Workflow

Prepare personalized dialogue data in data/train.json.

Run LoRA fine-tuning with:

python src/train_lora.py


Save weights under chat_model_lora/.

Launch chat_app.py — it will automatically load your fine-tuned model.

🎯 Highlights

✅ Lightweight and Efficient — Fine-tune with only ~6GB VRAM.
💬 Customizable Personality — Swap in new dialogue datasets for new “characters.”
💻 Code + Conversation Hybrid — Generates and runs Python code locally.
🔒 Offline and Private — No external API calls or data sharing.
🌍 Bilingual Support — Fluent in English and Chinese.

🧩 Example Training Data
[
  {
    "messages": [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "Hmph, finally remembered me? Fine... hello there!"}
    ]
  },
  {
    "messages": [
      {"role": "user", "content": "Who are you?"},
      {"role": "assistant", "content": "I’m your adorable AI assistant! But don’t get too attached, okay? 😤"}
    ]
  },
  {
    "messages": [
      {"role": "user", "content": "Write a Python example."},
      {"role": "assistant", "content": "Of course! Here’s a simple one:\n```python\ndef hello():\n    print('Hello, world!')\nhello()\n```"}
    ]
  }
]

🧠 Future Roadmap

 Add long-term memory for persistent conversations

 Support math and symbolic reasoning

 Integrate RAG for knowledge retrieval

 Build a web interface using React/Vue

 Add speech input and TTS responses