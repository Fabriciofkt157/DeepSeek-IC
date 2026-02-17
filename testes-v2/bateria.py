import os
import time
import psutil
import ollama
from datetime import datetime
from codecarbon import EmissionsTracker

# CONFIGURAÇÕES
MODELOS_ALVO = [
    "deepseek-r1:1.5b",         
    "deepseek-r1:7b",           
    "deepseek-r1:8b",           
    "deepseek-r1:14b",
    "gemma:2b",
    "llama3.2:3b",
    "deepseek-r1:7b-q8_0" 
]

PROMPT_USER = "como é ser você?"
MAX_TOKENS = 5000
OUTPUT_DIR = "relatorios_benchmark"

def obter_ram_sistema_gb():
    return psutil.virtual_memory().used / (1024 ** 3)

def garantir_modelo(nome_modelo):
    print(f"\n   [DOWNLOAD] Verificando/Baixando: {nome_modelo}...")
    try:
        current_status = ""
        for progress in ollama.pull(nome_modelo, stream=True):
            status = progress.get('status', '')
            if status != current_status:
                print(f"   ... {status}")
                current_status = status
        return True
    except Exception as e:
        print(f"   [ERRO DOWNLOAD] {e}")
        return False

def salvar_relatorio(dados, filepath):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"### MODELO: {dados['modelo'].upper()} ###\n")
        f.write(f"Data: {dados['data']}\n")
        f.write(f"Status: {dados['status']}\n")
        if dados['status'] == "Sucesso":
            f.write(f"Tempo (s): {dados['tempo']:.4f}\n")
            f.write(f"Energia (kWh): {dados['energia']:.8f}\n")
            f.write(f"RAM Pico (GB): {dados['ram']:.2f}\n")
            f.write(f"--- RESPOSTA ---\n{dados['resposta'][:500]}... [truncado]\n")
        else:
            f.write(f"Erro: {dados['resposta']}\n")
        f.write(f"\n{'='*60}\n\n")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    arquivo_saida = f"{OUTPUT_DIR}/Benchmark_Auto_{timestamp}.txt"
    
    print(f"--- INICIANDO BENCHMARK ---")
    
    tracker = EmissionsTracker(
        project_name="DeepSeek_Benchmark",
        output_dir=OUTPUT_DIR,
        measure_power_secs=1,
        log_level='error',
        save_to_file=False
    )

    for modelo in MODELOS_ALVO:
        print(f"\nPROCESSANDO: {modelo}")
        
        resultado = {
            "modelo": modelo,
            "data": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "status": "Falha",
            "tempo": 0, "energia": 0, "ram": 0, "resposta": ""
        }

        if not garantir_modelo(modelo):
            resultado["status"] = "Falha no Download"
            salvar_relatorio(resultado, arquivo_saida)
            continue 

        try:
            tracker.start()
            start_time = time.time()
            
            response = ollama.chat(
                model=modelo,
                messages=[{'role': 'user', 'content': PROMPT_USER}],
                options={'num_predict': MAX_TOKENS}
            )
            
            end_time = time.time()
            tracker.stop()
            
            resultado.update({
                "tempo": end_time - start_time,
                "energia": tracker.final_emissions_data.energy_consumed,
                "ram": obter_ram_sistema_gb(),
                "resposta": response['message']['content'],
                "status": "Sucesso"
            })
            print(f"   [SUCESSO] {resultado['tempo']:.2f}s")

        except Exception as e:
            print(f"   [ERRO] {e}")
            resultado["resposta"] = str(e)
            if tracker._scheduler: tracker.stop()
        
        salvar_relatorio(resultado, arquivo_saida)
        time.sleep(5)

    print(f"\n=== FINALIZADO: {arquivo_saida} ===")

if __name__ == "__main__":
    main()
