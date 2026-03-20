import numpy as np
import random
import matplotlib.pyplot as plt

#Ao adicionar perfis de investidor, o que muda é a função de fitness
perfis = {
    "conservador": {
        "retorno": 0.3,
        "dividendos": 0.2,
        "risco": 0.5
    },
    "dividendos": {
        "retorno": 0.2,
        "dividendos": 0.6,
        "risco": 0.2
    },
    "crescimento": {
        "retorno": 0.6,
        "dividendos": 0.1,
        "risco": 0.3
    }
}

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
def fitness(carteira, perfil):
    retorno = np.sum(carteira * retornos)
    dividendos_total = np.sum(carteira * dividendos)
    risco = np.sum(carteira * riscos)

    score = (
        perfil["retorno"] * retorno +
        perfil["dividendos"] * dividendos_total - 
        perfil["risco"] * risco
    )

    return score

#seleção por torneio
def selecao(populacao, perfil, k=3):
    candidatos = random.sample(populacao, k)
    candidatos.sort(key=lambda c: fitness(c, perfil), reverse=True)

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
def algoritmo_genetico(perfil):
    TAM_POP = 100
    GERACOES = 200
    ELITE_SIZE = 3

    populacao = criar_populacao(TAM_POP)

    historico_fitness = []
    historico_media = []

    for geracao in range(GERACOES):
        #Avaliação
        fitness_pop = [fitness(c, perfil) for c in populacao]
        melhor_fitness = max(fitness_pop)
        media_fitness = sum(fitness_pop) / len(fitness_pop)
        
        historico_fitness.append(melhor_fitness)
        historico_media.append(media_fitness)

        nova_pop = []

        #Se ELITE_SIZE for zero, simplesmente roda algoritmo genetico com nova_pop vazia
        if ELITE_SIZE > 0:
            #Ordena população (melhores primeiro)
            populacao_ordenada = sorted(populacao, key=lambda c: fitness(c, perfil), reverse=True)
            #Elitismo (copiando os melhores para nova_pop)
            nova_pop.extend(populacao_ordenada[:ELITE_SIZE])
        
        while len(nova_pop) < TAM_POP:
            pai1 = selecao(populacao, perfil)
            pai2 = selecao(populacao, perfil)

            filho = cruzamento(pai1, pai2)
            filho = mutacao(filho)

            nova_pop.append(filho)

        populacao = nova_pop

    melhor = max(populacao, key=lambda c: fitness(c, perfil))

    return melhor, historico_fitness, historico_media

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

#Rodar todos os perfis automaticamente
for nome, perfil in perfis.items():
    print("\n==========================")
    print("Perfil:", nome)

    melhor_carteira, hist_fitness, _ = algoritmo_genetico(perfil)

    for ativo, peso in zip(ativos, melhor_carteira):
        print(f"{ativo}: {peso:.2%}")

    retorno = round(np.sum(melhor_carteira * retornos), 3)
    dividendos_total = round(np.sum(melhor_carteira * dividendos), 3)
    risco = round(np.sum(melhor_carteira * riscos), 3)

    print("Retorno:", retorno)
    print("Dividendos:", dividendos_total)
    print("Risco:", risco)

plt.figure(figsize=(8, 4))

plt.plot(hist_fitness, label="Melhor Fitness")

plt.xlabel("Geração")
plt.ylabel("Fitness")
plt.title("Convergência do Algoritmo Genético")

plt.legend()
plt.savefig("com_elitismo.png")
plt.show()