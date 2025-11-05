from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

# ======== 配置 ========
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_DIR = "./chat_model_lora"
OUTPUT_DIR = "./chat_model_merged"

# ======== 检查路径 ========
if not os.path.exists(LORA_DIR):
    raise FileNotFoundError(f"❌ LoRA 文件夹未找到: {LORA_DIR}")

print("🔹 正在加载基础模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("🔹 正在加载 LoRA 适配器...")
model = PeftModel.from_pretrained(base_model, LORA_DIR)

print("🔹 开始合并 LoRA 权重（这可能需要几分钟）...")
merged_model = model.merge_and_unload()
merged_model.save_pretrained(OUTPUT_DIR)
print(f"✅ 合并完成！新模型已保存至: {OUTPUT_DIR}")

# 保存 tokenizer
print("💾 正在保存 tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Tokenizer 保存成功。")
