# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf", cache_dir="/root/autodl-tmp/huggingface/hub")
model = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf", cache_dir="/root/autodl-tmp/huggingface/hub")
