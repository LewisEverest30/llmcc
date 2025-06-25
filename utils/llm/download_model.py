from huggingface_hub import snapshot_download

''' llama2 7b
repo_id = "meta-llama/Llama-2-7b-hf"  # 模型在 Hugging Face 上的名称
local_dir = "./downloaded_plms/llama2-7b"  # 本地模型存储的地址
'''

'''
repo_id = "openai-community/gpt2"  # 模型在 Hugging Face 上的名称
local_dir = "./downloaded_plms/gpt2"  # 本地模型存储的地址
'''

repo_id = "Qwen/Qwen2-7B"  # 模型在 Hugging Face 上的名称
local_dir = "plm/qwen"  # 本地模型存储的地址


token = "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"  # 您在 Hugging Face 上生成的 access token

proxies = {
    'http': "http://127.0.0.1:7890",
    'https': "http://127.0.0.1:7890"
}

snapshot_download(repo_id=repo_id, 
                  local_dir=local_dir, 
                  use_auth_token=token,
                  proxies=proxies
                  )