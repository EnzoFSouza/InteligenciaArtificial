import numpy as np
import random

ativos = ["PETR4", "VALE3", "ITUB4", "WEGE3", "BBAS3"]

#dados simulados
retornos = np.array([0.15, 0.12, 0.10, 0.18, 0.11]) #retorno esperado
dividendos = np.array([0.10, 0.07, 0.08, 0.02, 0.09]) #dividend yield
riscos = np.array([0.20, 0.18, 0.12, 0.25, 0.14]) #risco individual
#risco --> desvio padrão dos retornos OU volatilidade --> Quanto MAIOR, PIOR

NUM_ATIVOS = len(ativos)
PESO_MIN = 0.05 #peso mínimo que um ativo pode ter em uma carteira, neste caso um ativo ocupa no mínimo 5%
PESO_MAX = 0.35 #peso máximo que um ativo pode ter em uma carteira, neste caso um ativo ocupa no máximo 35%

#criar carteira aleatória (os pesos precisam somar 1)
def criar_carteira():
    pesos = np.random.rand(NUM_ATIVOS)
    return ajustar_carteira(pesos)

#criar população inicial
def criar_populacao(tamanho):
    return [criar_carteira() for _ in range(tamanho)]

#fitness --> maximizar retorno e dividendos e minimizar risco
def fitness(carteira):
    retorno = np.sum(carteira * retornos)
    dividendos_total = np.sum(carteira * dividendos)
    risco = np.sum(carteira * riscos)

    score = retorno + dividendos_total - risco

    return score

#seleção por torneio
def selecao(populacao, k=3):
    candidatos = random.sample(populacao, k)
    candidatos.sort(key=fitness, reverse=True)

    return candidatos[0]

#cruzamento
def cruzamento(pai1, pai2):
    ponto = random.randint(1, NUM_ATIVOS-1)
    filho = np.concatenate((pai1[:ponto], pai2[ponto:]))
    return ajustar_carteira(filho)

#mutação --> pequena alteração nos pesos
def mutacao(carteira, taxa=0.1):
    if random.random() < taxa:
        i = random.randint(0, NUM_ATIVOS-1)
        carteira[i] += np.random.normal(0, 0.05)
        carteira = ajustar_carteira(carteira)

    return carteira

#loop principal do algoritmo
def algoritmo_genetico():
    TAM_POP = 100
    GERACOES = 200

    populacao = criar_populacao(TAM_POP)
    for geracao in range(GERACOES):
        nova_pop = []

        for _ in range(TAM_POP):
            pai1 = selecao(populacao)
            pai2 = selecao(populacao)

            filho = cruzamento(pai1, pai2)
            filho = mutacao(filho)

            nova_pop.append(filho)

        populacao = nova_pop

    melhor = max(populacao, key=fitness)

    return melhor

#Fazer com que a carteira respeite PESO_MIN e PESO_MAX
def ajustar_carteira(pesos):
    # aplica limites mínimo e máximo
    pesos = np.clip(pesos, PESO_MIN, PESO_MAX)
    # normaliza
    pesos = pesos / np.sum(pesos)
    # garante novamente limites após normalização
    for i in range(len(pesos)):
        if pesos[i] > PESO_MAX:
            pesos[i] = PESO_MAX
        if pesos[i] < PESO_MIN:
            pesos[i] = PESO_MIN

    # normaliza novamente
    pesos = pesos / np.sum(pesos)
    return pesos

melhor_carteira = algoritmo_genetico()
retorno = np.sum(melhor_carteira * retornos)
dividendos_total = np.sum(melhor_carteira * dividendos)
risco = np.sum(melhor_carteira * riscos)

print("Melhor carteira encontrada:\n")
for ativo, peso in zip(ativos, melhor_carteira):
    print(f"{ativo}: {peso:.2%}")
print("\nFitness:", fitness(melhor_carteira))
print("Retorno Esperado:", retorno)
print("Dividendos Total:", dividendos_total)
print("Risco:", risco)