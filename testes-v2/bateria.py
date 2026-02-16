import os
import time
import psutil
import ollama
import getpass
from datetime import datetime
from codecarbon import EmissionsTracker
from huggingface_hub import login


# COLOQUE SEU TOKEN DENTRO DAS ASPAS ABAIXO:
SEU_TOKEN_HF = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 


def autenticar_huggingface():
    print("\n--- Autenticação Hugging Face ---")
    
    # 1. Tenta usar o token fixo no código
    if SEU_TOKEN_HF and SEU_TOKEN_HF.strip() != "" and "hf_" in SEU_TOKEN_HF:
        try:
            print("Tentando autenticar com o token inserido no código...")
            login(token=SEU_TOKEN_HF)
            print(">> Autenticação realizada com SUCESSO via token fixo!")
            return # Sai da função, tudo certo
        except Exception as e:
            print(f"[ERRO] O token no código é inválido: {e}")
            print("Alternando para inserção manual...")

    try:
        # Verifica variáveis de ambiente
        token_env = os.getenv("HF_TOKEN")
        if token_env:
            print("Usando token das variáveis de ambiente...")
            login(token=token_env)
        else:
            # Pede ao usuário
            print("Nenhum token fixo encontrado.")
            token_input = getpass.getpass("Insira seu token HF (Write) manualmente: ")
            if token_input.strip():
                login(token=token_input)
                print(">> Autenticação manual realizada com SUCESSO!")
            else:
                print(">> Continuando sem autenticação (apenas modelos públicos).")
    except Exception as e:
        print(f"Erro na autenticação: {e}")

MODELOS_ALVO = [
    
    "deepseek-r1:1.5b",         # Qwen Distill (Leve)
    "deepseek-r1:7b",           # Llama Distill (Padrão)
    "deepseek-r1:8b",           # Llama Distill (Intermediário)
    "deepseek-r1:14b",          # Qwen Distill (Médio - Requer ~8GB VRAM)
    "deepseek-r1:32b",          # Qwen Distill (Pesado - Requer ~20GB VRAM)
    
    # --- DeepSeek V3/Coder (Base) ---
    "deepseek-coder:6.7b",
    
    
    "deepseek-r1:7b-q8_0",      # Alta precisão (8-bit)
    "deepseek-r1:7b-q4_K_M",    # Balanceado (4-bit)
    "deepseek-r1:7b-fp16",      # Precisão total (16-bit - PESADO)
    
    # --- Outros Modelos para Comparação ---
    "gemma:2b",
    "llama3.2:3b",
    "mistral:7b"
]

PROMPT_USER = "como é ser você?"
MAX_TOKENS = 5000
OUTPUT_DIR = "relatorios_benchmark"

def obter_ram_sistema_gb():
    """Retorna o uso total de RAM do sistema em GB."""
    return psutil.virtual_memory().used / (1024 ** 3)

def salvar_relatorio(dados, filepath):
    """Escreve os dados no arquivo TXT de forma incremental."""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"### MODELO: {dados['modelo'].upper()} ###\n")
        f.write(f"Data/Hora: {dados['data']}\n")
        f.write(f"Tempo de Execução (s): {dados['tempo']:.4f}\n")
        f.write(f"Consumo Energia (kWh): {dados['energia']:.8f}\n")
        f.write(f"RAM Sistema Final (GB): {dados['ram']:.2f}\n")
        f.write(f"Status: {dados['status']}\n")
        f.write(f"--- RESPOSTA GERADA ---\n")
        f.write(f"{dados['resposta']}\n")
        f.write(f"\n{'='*60}\n\n")


def main():

    autenticar_huggingface()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Cria nome do arquivo com timestamp único
    timestamp_global = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    arquivo_saida = f"{OUTPUT_DIR}/Benchmark_DeepSeek_{timestamp_global}.txt"
    
    print(f"\n=== INICIANDO BENCHMARK DE {len(MODELOS_ALVO)} MODELOS ===")
    print(f"Relatório será salvo em: {arquivo_saida}")
    
    # Cabeçalho do arquivo
    with open(arquivo_saida, "w", encoding="utf-8") as f:
        f.write(f"RELATÓRIO DE BENCHMARK DE LLMs\n")
        f.write(f"Início: {timestamp_global}\n")
        f.write(f"Prompt: '{PROMPT_USER}'\n")
        f.write(f"Limite Tokens: {MAX_TOKENS}\n")
        f.write("="*60 + "\n\n")

    # Configura CodeCarbon (rastreamento de energia)
    # log_level='error' para não poluir o terminal
    tracker = EmissionsTracker(
        project_name="DeepSeek_Benchmark", 
        output_dir=OUTPUT_DIR,
        measure_power_secs=0.5,
        log_level='error',
        save_to_file=False 
    )

    for modelo in MODELOS_ALVO:
        print(f"\n>> Preparando modelo: {modelo}...")
        
        resultado = {
            "modelo": modelo,
            "data": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "tempo": 0,
            "energia": 0,
            "ram": 0,
            "resposta": "",
            "status": "Sucesso"
        }

        try:
            # Tenta puxar o modelo se não existir (pode demorar)
            try:
                # Verificação rápida se o modelo está carregado (opcional, o chat faz o pull)
                pass 
            except Exception:
                print(f"   Aviso: Modelo {modelo} pode precisar de download...")

            # --- INÍCIO DA MEDIÇÃO ---
            ram_antes = obter_ram_sistema_gb()
            tracker.start()
            start_time = time.time()
            
            # Chamada ao modelo
            # num_predict define o limite de tokens de saída
            response = ollama.chat(
                model=modelo, 
                messages=[{'role': 'user', 'content': PROMPT_USER}],
                options={'num_predict': MAX_TOKENS} 
            )
            
            end_time = time.time()
            emissions = tracker.stop()
            # --- FIM DA MEDIÇÃO ---

            # Preenchendo dados
            resultado["tempo"] = end_time - start_time
            resultado["energia"] = tracker.final_emissions_data.energy_consumed # kWh
            resultado["ram"] = obter_ram_sistema_gb() # RAM usada durante/após a carga
            resultado["resposta"] = response['message']['content']
            
            print(f"   [OK] Concluído em {resultado['tempo']:.2f}s | Energia: {resultado['energia']:.6f} kWh")

        except ollama.ResponseError as e:
            print(f"   [ERRO] Falha no Ollama: {e}")
            resultado["status"] = f"ERRO API: {e}"
            resultado["resposta"] = "Nenhuma resposta gerada devido a erro."
            if tracker._scheduler: tracker.stop()
            
        except Exception as e:
            print(f"   [ERRO] Falha Geral: {e}")
            resultado["status"] = f"ERRO CRÍTICO: {e}"
            if tracker._scheduler: tracker.stop()

        # Salva imediatamente no arquivo (append) para não perder dados se o script quebrar
        salvar_relatorio(resultado, arquivo_saida)
        
        # Pausa para resfriamento e limpeza de memória
        print("   Resfriando por 5 segundos...")
        time.sleep(5)

    print(f"\n=== BENCHMARK FINALIZADO ===")
    print(f"Verifique o arquivo: {arquivo_saida}")

if __name__ == "__main__":
    main()
