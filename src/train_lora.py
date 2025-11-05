from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import os

# =========================================================
# ⚙️ 模型配置
# =========================================================
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./chat_model_lora"
DATA_PATH = "./data/train.json"

# =========================================================
# 🧠 加载 Tokenizer & Dataset
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
dataset = load_dataset("json", data_files=DATA_PATH)

def preprocess(example):
    # 多轮上下文拼接：让模型知道完整对话，而不仅仅是最后一句
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=1024,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

# =========================================================
# 🧩 LoRA 配置
# =========================================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# =========================================================
# ⚙️ 模型加载与 LoRA 注入
# =========================================================
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================================================
# 🧾 训练参数设置
# =========================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none"
)

# =========================================================
# 🚀 训练启动
# =========================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()

# =========================================================
# 💾 保存模型
# =========================================================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ LoRA 多轮对话模型训练完成，已保存到:", OUTPUT_DIR)
