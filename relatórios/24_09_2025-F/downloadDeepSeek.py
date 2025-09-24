from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Verificar se temos GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Baixar e carregar modelo
model_name = "deepseek-ai/deepseek-llm-1.3b"

print("Baixando modelo...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Usar metade da precisão
    device_map="auto"
)

print("Modelo carregado! Testando...")

# Teste rápido
input_text = "Explique o que é inteligência artificial:"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Resposta: {response}")

