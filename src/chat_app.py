import sys
import os
import types
import platform
import io
import contextlib
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================================================
# 🚫 伪造 bitsandbytes 模块，防止 Windows CUDA 报错
# =========================================================
fake_bnb = types.ModuleType("bitsandbytes")
fake_bnb.__spec__ = types.SimpleNamespace()
fake_bnb.__file__ = "fake_bitsandbytes.py"
fake_bnb.__path__ = []
fake_bnb.nn = types.SimpleNamespace(modules=types.SimpleNamespace(Linear8bitLt=None), Linear8bitLt=None)
fake_bnb.cuda_setup = types.SimpleNamespace(main=lambda: None)
sys.modules["bitsandbytes"] = fake_bnb

# =========================================================
# 🌍 环境设置
# =========================================================
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["BITSANDBYTES_DISABLE"] = "1"
os.environ["PEFT_BACKEND"] = "TORCH"
os.environ["USE_TORCH_FOR_LORA"] = "1"

system = platform.system()
print(f"🖥️ 当前系统：{system}")
if system == "Windows":
    print("⚠️ 自动禁用 bitsandbytes（Windows 不支持 CUDA DLL）")
else:
    print("✅ Linux / macOS 可使用量化")

# =========================================================
# 📦 模型路径配置
# =========================================================
BASE_MODEL = "./chat_model_merged"      # ✅ 已合并好的模型
FINETUNED_MODEL = "./chat_model_lora"   # 若无 LoRA，可设为 None

# =========================================================
# 🧠 加载 Tokenizer
# =========================================================
print("🔹 正在加载 tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

if not getattr(tokenizer, "chat_template", None):
    print("⚠️ chat_template 丢失，自动补充默认模板。")
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )

# =========================================================
# ⚙️ 加载模型
# =========================================================
print("🔹 正在加载基础模型 ...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True
)
print("✅ 基础模型加载完成")

# =========================================================
# 🔧 尝试加载 LoRA（若存在）
# =========================================================
if FINETUNED_MODEL is not None and os.path.exists(FINETUNED_MODEL):
    print("🔹 检测到 LoRA 模型，正在加载并合并 ...")
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)
        model = model.merge_and_unload()
        print("✅ LoRA 权重已成功合并！")
    except Exception as e:
        print(f"⚠️ LoRA 加载失败，将继续使用基础模型: {e}")
        model = base_model
else:
    print("🧩 未指定 LoRA，直接使用合并模型。")
    model = base_model

# =========================================================
# 🧩 Python 代码执行模块
# =========================================================
def execute_python(code):
    """安全执行 Python 代码，并捕获输出"""
    try:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(code, {})
        output = buffer.getvalue()
        if not output.strip():
            output = "✅ 代码执行成功（无输出）"
        return output
    except Exception as e:
        return f"⚠️ 执行出错：{e}"

# =========================================================
# 💬 聊天逻辑：可自由生成 + 自动检测代码
# =========================================================
def chat_fn(message, history):
    try:
        messages = []
        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})
        messages.append({"role": "user", "content": message})

        # === Step 0. 额外系统提示词：强制启发模型写代码 ===
        sys_prompt = (
            "你是一个会写Python代码的AI助手。如果用户请求'写代码'、'帮我写Python函数'或类似内容，"
            "请返回带有 ```python ... ``` 的完整代码块，不要省略。"
            "生成后，代码将会被自动执行，请确保能直接运行。"
        )
        messages.insert(0, {"role": "system", "content": sys_prompt})

        # 生成 prompt
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # ======= Step 1. 模型生成 =======
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,  # 降低随机性，让输出更像代码
                top_p=0.9,
                repetition_penalty=1.05,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = text[len(prompt):].strip() if prompt else text.strip()

        # ======= Step 2. 检测是否包含Python代码 =======
        msg_lower = message.lower()
        if "python" in msg_lower or "代码" in message:
            code_start = answer.find("```python")
            code_end = answer.find("```", code_start + 9)
            if code_start != -1 and code_end != -1:
                code = answer[code_start + 9:code_end].strip()
                result = execute_python(code)
                answer += f"\n\n🧠 执行结果：\n{result}"
            else:
                answer += "\n\n⚠️ 没检测到可执行的Python代码块，请重试。"

        # ======= Step 3. 傲娇润色 =======
        if any(x in msg_lower for x in ["你好", "hi", "hello", "hey"]):
            answer = f"才、才不是特地想理你呢……你好呀！😤 {answer}"
        elif "谁" in message or "who" in msg_lower:
            answer = f"哼～我当然是你的AI小帮手啦，不过别太依赖我哦～ {answer}"
        elif "干嘛" in message or "doing" in msg_lower:
            answer = f"才、才没在想你啦！我在等你问我问题呢～ {answer}"
        elif len(answer) < 6:
            answer = f"哼？你说的我不太懂呢，再说一遍嘛～ {answer}"

        answer = answer.replace("。", "～")

        print(f"\n🗨️ User: {message}\n💬 Bot: {answer}\n{'-'*40}")
        return answer

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"⚠️ 出错：{e}"


# =========================================================
# 🚀 启动 Gradio 聊天界面
# =========================================================
print("🚀 启动聊天界面中...")
gr.ChatInterface(
    fn=chat_fn,
    title="💬 傲娇AI TsundereBot (会写代码的版本)",
    description="支持 LoRA / 合并模型 | 会生成和执行 Python 代码 | 双语聊天 | 本地运行 💻",
    theme="soft",
).launch(server_name="127.0.0.1", server_port=7860)
