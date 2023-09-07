"""Script used to save huggingface models to the docker container."""

from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Read huggingface access token as a docker secret
access_token = Path("/run/secrets/hf_token").read_text().strip()

# Download llama-2-7b-chat-hf
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    token=access_token,
)
hf_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    low_cpu_mem_usage=True,
    token=access_token,
)
