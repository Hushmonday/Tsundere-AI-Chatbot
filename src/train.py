from datasets import load_dataset
from bitsandbytes.optim import Adam8bit
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling  # ✅ 新增
)
import torch

# =========================================================
# 🧠 模型与数据集加载
# =========================================================
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
dataset = load_dataset("json", data_files="data/train.json")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
model.gradient_checkpointing_enable()  # ✅ 节省一半激活显存


# =========================================================
# 🔧 数据预处理
# =========================================================
def preprocess(example):
    # 将 messages 转成单一字符串，并自动加上特殊标记
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    out = tokenizer(
        text,
        truncation=True,
        max_length=512,   # ✅ 从1024降到512
        padding="max_length"
    )

    out["labels"] = out["input_ids"].copy()
    return out


tokenized = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

# =========================================================
# ⚙️ 训练参数配置
# =========================================================
args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,  # ✅ 自动检测是否支持 CUDA
    logging_steps=10,
    save_strategy="epoch",
    remove_unused_columns=False,

    optim="adamw_bnb_8bit",   # ✅ 用 8-bit 优化器代替 AdamW
    fp16=False,
    bf16=True,
    per_device_train_batch_size=1,
)

# =========================================================
# 🧩 修复 dtype 报错：定义 DataCollator
# =========================================================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # ✅ 自回归语言模型必须设为 False
)

# =========================================================
# 🚀 启动 Trainer
# =========================================================
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    data_collator=data_collator
)


trainer.train()

# =========================================================
# 💾 保存模型与分词器
# =========================================================
trainer.save_model("./chat_model")
tokenizer.save_pretrained("./chat_model")

print("✅ 训练完成！模型已保存到 ./chat_model")
