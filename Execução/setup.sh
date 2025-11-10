#!/bin/bash

# --- 1. Definições Iniciais ---
ENV_NAME="deepseek_teste"
PYTHON_SCRIPT="deepseek_inference.py"


# --- 2. Criação do Ambiente Virtual (venv) ---
echo "1. Criando ambiente virtual Python chamado '$ENV_NAME'..."
python3 -m venv $ENV_NAME

if [ $? -ne 0 ]; then
    echo "ERRO: Falha ao criar o ambiente virtual. Certifique-se de que 'python3' está instalado."
    exit 1
fi

echo "   Ambiente virtual criado com sucesso."
echo " "

# --- 3. Ativação do Ambiente Virtual ---
echo "2. Ativando o ambiente virtual..."
source $ENV_NAME/bin/activate
echo "   Ambiente virtual ativado."
echo " "

# --- 4. Instalação das Dependências (pip) ---
echo "3. Instalando as bibliotecas necessárias: transformers, torch, accelerate, bitsandbytes..."
pip install -q transformers torch accelerate bitsandbytes

if [ $? -ne 0 ]; then
    echo "ERRO: Falha ao instalar as dependências. Verifique sua conexão com a internet."
    deactivate
    exit 1
fi

echo "   Dependências instaladas com sucesso."
