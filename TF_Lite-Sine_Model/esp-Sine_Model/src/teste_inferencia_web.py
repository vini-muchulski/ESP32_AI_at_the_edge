#!/usr/bin/env python3
"""
Cliente Python para testar a API de seno do ESP32
"""

import requests
import json

# Configure o IP do seu ESP32 aqui
ESP32_IP = "192.168.0.119"  # Substitua pelo IP real do seu ESP32
BASE_URL = f"http://{ESP32_IP}"

def testar_seno(angulo):
    """Testa o cálculo de seno para um ângulo específico"""
    try:
        url = f"{BASE_URL}/seno"
        params = {"angulo": angulo}
        
        print(f"Testando sin({angulo}°)...")
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Resposta: sin({data['angulo_graus']}°) = {data['seno']:.6f}")
            return data
        else:
            print(f"✗ Erro HTTP {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Erro de conexão: {e}")
        return None

def main():
    print("=== Teste da API ESP32 - Cálculo de Seno ===\n")
    
    # Ângulos de teste
    angulos_teste = [ 90,30,45]
    
    print(f"Conectando ao ESP32 em: {BASE_URL}\n")
    
    # Testa cada ângulo
    sucessos = 0
    for angulo in angulos_teste:
        resultado = testar_seno(angulo)
        if resultado:
            sucessos += 1
        print()  # Linha em branco
    
    print(f"Teste concluído: {sucessos}/{len(angulos_teste)} sucessos")

if __name__ == "__main__":
    main()