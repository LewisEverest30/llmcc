import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

qwen = AutoModelForCausalLM.from_pretrained("plm_model\qwen2\small").to('cuda')

print("type of qwen: ", type(qwen))
random_tensor = torch.randn(1, 1, 896).to('cuda')
# print(random_tensor)

qwen.set_input_embeddings(None)

# outputs = qwen(inputs_embeds=random_tensor)

help(qwen.__call__)

pass