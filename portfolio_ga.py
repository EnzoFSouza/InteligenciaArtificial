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
    #np.random.rand(n) --> gera array com n números aleatórios entre 0 e 1
    pesos = np.random.rand(NUM_ATIVOS) 

    #transforma o vetor aleatório em uma carteira válida e retorna
    return ajustar_carteira(pesos)

#criar população inicial de indivíduos --> Lista de carteiras
def criar_populacao(tamanho):
    #list comprehension
    #[criar_carteira() for _ in range(tamanho)] é equivalente a:
    #populacao = []
    #for i in range(tamanho):
    #    populacao.append(criar_carteira())

    #Cada indivíduo --> Uma carteira
    return [criar_carteira() for _ in range(tamanho)]

#fitness --> maximizar retorno e dividendos e minimizar risco
def fitness(carteira, perfil, penalizacao = True):
    #carteira --> array com pesos [0.1, 0.2, ...]
    #perfil --> dicionario com pesos
    #penalizacao --> liga/desliga penalizacao

    #retorno total da carteira = soma(peso_ativo_n * retorno_ativo_n)
    retorno = np.sum(carteira * retornos)
    dividendos_total = np.sum(carteira * dividendos)
    risco = np.sum(carteira * riscos)
    
    #Maximizar retorno e dividendos (+) e minimizar risco (-)
    score = (
        perfil["retorno"] * retorno +
        perfil["dividendos"] * dividendos_total - 
        perfil["risco"] * risco
    )

    #adicionando penalização por concentração
    #sem isso, o algoritmo genético tende a concentrar valores
    #em dois ativos e investir o mínimo de recursos nos ativos restantes
    if penalizacao:
        #carteira distribuida
        #[0.1, 0.1, ..., 0.1] --> Soma dos quadrados: 0.1

        #carteira concentrada
        #[0.9, 0.0.1, ..., 0.01] --> Soma dos quadrados: 0.82
        concentracao = np.sum(carteira**2)

        #penalização no score
        score -= 0.2 * concentracao
        
    return score

#seleção por torneio
def selecao(populacao, perfil, penalizacao, k=3):
    #escolhe k carteiras aleatórias da lista, sem repetição
    candidatos = random.sample(populacao, k)

    #.sort() --> ordena a lista in-place --> modifica a propria lista
    #key --> define como comparar os elementos --> comparando pelo fitness
    #lambda c: fitness(c, perfil) função anônima que equivale a:
    #def f(c):
    #   return fitness(c, perfil)
    
    #calcula o fitness de cada carteira --> Python precisa chamar a função pra cada elemento
    #fitness(c1), fitness(c2), fitness(c3) e ordena com base nesses valores

    #reverse = True --> Ordena do maior pro menor --> Melhor fitness fica em primeiro
    candidatos.sort(key=lambda c: fitness(c, perfil, penalizacao), reverse=True)

    #retorna o melhor dos k candidatos escolhidos
    return candidatos[0]

#cruzamento
def cruzamento(pai1, pai2):
    #random.randint(1, NUM_ATIVOS-1) --> gera um número inteiro aleatório entre 1 e NUM_ATIVOS-1
    #inclui os dois extremos
    ponto = random.randint(1, NUM_ATIVOS-1)

    #Fatiamento de array / Slicing
    #pai1[:ponto] --> pega do início até ponto - 1
    #pai2[ponto:] --> pega do ponto até o final

    #np.concatenate --> junta arrays
    #Obs.: precisa passar uma tupla como parâmetro: (array1, array2)
    filho = np.concatenate((pai1[:ponto], pai2[ponto:]))

    #pai1 = [A, A, A, A, A, A]
    #pai2 = [B, B, B, B, B, B]
    #ponto = 3
    #pai1[:3] --> [A, A, A]
    #pai2[3:] --> [B, B, B]
    #filho = [A, A, A, B, B, B]
    
    #cruzamento do tipo crossover

    #ajuste da carteira para estar de acordo com restrições
    return ajustar_carteira(filho)

#mutação --> pequena alteração nos pesos
#introduz novidade no sistema
def mutacao(carteira, taxa=0.1):
    #random.random() --> gera número aleatório entre 0 e 1
    #como taxa = 0.1, 10% de chance de mutação
    #90% das vezes nada ocorre
    #10% das vezes mutação ocorre
    if random.random() < taxa:
        #escolha de índice aleatório da carteira
        i = random.randint(0, NUM_ATIVOS-1)

        #np.random.normal(0, 0.05) gera um número com distribuição normal (gaussiana)
        #média = 0, desvio_padrão = 0.05
        #para criar uma pequena variação, como +0.02, -0.01, +0.07, -0.03
        
        #altera o peso de um ativo na carteira
        carteira[i] += np.random.normal(0, 0.05)

        #ajuste da carteira para estar de acordo com restrições
        carteira = ajustar_carteira(carteira)

    return carteira

def algoritmo_genetico(perfil, elitismo = False, penalizacao = True):
    #perfil --> define o tipo de investidor
    #elitismo --> ativa ou desativa preservação dos melhores
    #penalização --> ativa ou desativa diversificação

    #Parâmetros do algoritmo
    TAM_POP = 100
    GERACOES = 200
    ELITE_SIZE = 3 if elitismo else 0

    #população inicial --> criar 100 carteira aleatórias
    populacao = criar_populacao(TAM_POP)

    #lista para guardar o melhor valor de cada geração --> utilizado nos gráficos
    historico_fitness = []

    #loop principal --> evolução
    #repete 200 vezes
    for geracao in range(GERACOES):
        #Avaliação da população
        #list comprehension
        #fitness_pop = [fitness(c, perfil, penalizacao) for c in populacao]
        #equivale a:
        #fitness_pop = []
        #for c in populacao:
        #   fitness_pop.append(fitness(c, perfil, penalizacao))

        #lista com os scores
        fitness_pop = [
            fitness(c, perfil, penalizacao) for c in populacao
        ]
        
        #guarda o melhor resultado da geração
        melhor_fitness = max(fitness_pop)
        historico_fitness.append(melhor_fitness)

        #construção da próxima geração
        nova_pop = []

        #Se ELITE_SIZE for zero, simplesmente roda algoritmo genetico com nova_pop vazia
        if ELITE_SIZE > 0:
            #sorted() --> retorna nova lista, não modifica original
            #Ordena população (melhores primeiro)
            populacao_ordenada = sorted(
                populacao,
                key=lambda c: fitness(c, perfil, penalizacao),
                reverse=True
            )
            #Elitismo (copiando os melhores para nova_pop)
            #populacao_ordenada[:ELITE_SIZE] --> pega os 3 melhores
            #extend --> adiciona vários elementos pra à lista
            nova_pop.extend(populacao_ordenada[:ELITE_SIZE])
        
        #gerar nova população, continua até ter 100 indivíduos
        while len(nova_pop) < TAM_POP:
            
            #escolhe dois pais
            pai1 = selecao(populacao, perfil, penalizacao)
            pai2 = selecao(populacao, perfil, penalizacao)

            #mistura os pais
            filho = cruzamento(pai1, pai2)

            #mutação aleatória
            filho = mutacao(filho)

            #adiciona filho à população
            nova_pop.append(filho)

        #atualiza população --> nova geração substitui a antiga
        populacao = nova_pop

    #pega a melhor carteira da última geração
    melhor = max(populacao, key=lambda c: fitness(c, perfil, penalizacao))

    return melhor, historico_fitness

#Fazer com que a carteira respeite PESO_MIN e PESO_MAX
def ajustar_carteira(pesos):
    # limita cada valor do array numpy pesos [0.2, 0.1, ...] entre um mínimo e um máximo
    pesos = np.clip(pesos, PESO_MIN, PESO_MAX)
    
    # normaliza --> Faz a soma dos pesos virar 100%
    pesos = pesos / np.sum(pesos)
    
    # garante novamente limites após normalização
    for i in range(len(pesos)):
        if pesos[i] > PESO_MAX:
            pesos[i] = PESO_MAX
        if pesos[i] < PESO_MIN:
            pesos[i] = PESO_MIN

    # normaliza novamente (necessário pois valores podem ser alterados no for)
    pesos = pesos / np.sum(pesos)

    #retorna o array ajustado
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