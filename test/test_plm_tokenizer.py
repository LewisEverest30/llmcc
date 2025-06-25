import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# 加载与指定模型名称对应的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("plm_model\qwen2\small")


# 使用 tokenizer 对文本进行编码
# encoded_input = tokenizer("135 135 135 135 135 135")
# encoded_input = tokenizer.encode("2.0 0.0 6.013 10.0 0.0 6.037 5.0 0.0 6.06")

encoded_input = tokenizer("2.0 0.0 6.013 10.0 0.0 6.037 5.0 0.0 6.06")

print(encoded_input)