from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "gpt2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

print("nano-gpt terminal mode. Type exit to quit.")
while True:
    prompt = input("You> ").strip()
    if prompt.lower() in {"exit", "quit"}:
        break
    if not prompt:
        continue

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True, 
        temperature=0.9, 
        top_k=50, 
        top_p=0.95
    )
    print("GPT>", tokenizer.decode(output[0], skip_special_tokens=True))
