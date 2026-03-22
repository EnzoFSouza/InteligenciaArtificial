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

ativos = ["PETR4", "VALE3", "ITUB4", "WEGE3", "BBAS3", #ações
          "HGLG11", "XPLG11", "KNRI11", "MXRF11", "VISC11" #fiis
]

#dados simulados
retornos = np.array([0.15, 0.12, 0.10, 0.18, 0.11, #ações
                     0.08, 0.09, 0.07, 0.06, 0.085 #fiis
]) #retorno esperado

dividendos = np.array([0.10, 0.07, 0.08, 0.02, 0.09, #ações
                       0.11, 0.10, 0.09, 0.12, 0.095 #fiis
]) #dividend yield

riscos = np.array([0.20, 0.18, 0.12, 0.25, 0.14, #ações
                   0.10, 0.11, 0.09, 0.08, 0.10 #fiis
]) #risco individual

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
def fitness(carteira, perfil, penalizacao = True):
    retorno = np.sum(carteira * retornos)
    dividendos_total = np.sum(carteira * dividendos)
    risco = np.sum(carteira * riscos)
    
    score = (
        perfil["retorno"] * retorno +
        perfil["dividendos"] * dividendos_total - 
        perfil["risco"] * risco
    )

    #adicionando penalização por concentração
    #sem isso, o algoritmo genético tende a concentrar valores
    #em dois ativos e investir o mínimo de recursos nos ativos restantes
    if penalizacao:
        concentracao = np.sum(carteira**2)
        score -= 0.2 * concentracao
        
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
def algoritmo_genetico(perfil, elitismo = False, penalizacao = True):
    TAM_POP = 100
    GERACOES = 200
    ELITE_SIZE = 3 if elitismo else 0

    populacao = criar_populacao(TAM_POP)

    historico_fitness = []

    for geracao in range(GERACOES):
        #Avaliação
        fitness_pop = [
            fitness(c, perfil, penalizacao) for c in populacao
        ]

        melhor_fitness = max(fitness_pop)
        historico_fitness.append(melhor_fitness)

        nova_pop = []

        #Se ELITE_SIZE for zero, simplesmente roda algoritmo genetico com nova_pop vazia
        if ELITE_SIZE > 0:
            #Ordena população (melhores primeiro)
            populacao_ordenada = sorted(
                populacao,
                key=lambda c: fitness(c, perfil, penalizacao),
                reverse=True
            )
            #Elitismo (copiando os melhores para nova_pop)
            nova_pop.extend(populacao_ordenada[:ELITE_SIZE])
        
        while len(nova_pop) < TAM_POP:
            pai1 = selecao(populacao, perfil)
            pai2 = selecao(populacao, perfil)

            filho = cruzamento(pai1, pai2)
            filho = mutacao(filho)

            nova_pop.append(filho)

        populacao = nova_pop

    melhor = max(populacao, key=lambda c: fitness(c, perfil, penalizacao))

    return melhor, historico_fitness

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


#Gráfico 1 --> Perfis com elitismo
plt.figure(figsize=(8,5))

for nome, perfil in perfis.items():
    _, hist = algoritmo_genetico(perfil, elitismo=True, penalizacao=True)
    plt.plot(hist, label=nome)

plt.title("Comparação entre Perfis de Investidor")
plt.xlabel("Geração")
plt.ylabel("Fitness")
plt.legend()

plt.savefig("comparacao_perfis.png", dpi=300)
plt.show()

#Gráfico 2 --> Com vs Sem elitismo
perfil = perfis["dividendos"]

_, hist_sem = algoritmo_genetico(perfil, elitismo=False, penalizacao=True)
_, hist_com = algoritmo_genetico(perfil, elitismo=True, penalizacao=True)

plt.figure(figsize=(8,5))

plt.plot(hist_sem, label="Sem Elitismo")
plt.plot(hist_com, label="Com Elitismo")

plt.title("Impacto do Elitismo")
plt.xlabel("Geração")
plt.ylabel("Fitness")
plt.legend()

plt.savefig("elitismo.png", dpi=300)
plt.show()

#Gráfico 3 --> Penalização de concentração
perfil = perfis["dividendos"]

melhor_sem, hist_sem_penal = algoritmo_genetico(perfil, elitismo=True, penalizacao=False)
melhor_com, hist_com_penal = algoritmo_genetico(perfil, elitismo=True, penalizacao=True)

plt.figure(figsize=(8,5))

plt.plot(hist_sem_penal, label="Sem Penalização")
plt.plot(hist_com_penal, label="Com Penalização")

plt.title("Impacto da Penalização de Concentração")
plt.xlabel("Geração")
plt.ylabel("Fitness")
plt.legend()

plt.savefig("penalizacao.png", dpi=300)
plt.show()

#Comparação de Carteiras
print("\nSem penalização:")
for a, p in zip(ativos, melhor_sem):
    print(a, f"{p:.2%}")

print("\nCom penalização:")
for a, p in zip(ativos, melhor_com):
    print(a, f"{p:.2%}")