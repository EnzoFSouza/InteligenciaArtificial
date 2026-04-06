import numpy as np
import random
import matplotlib.pyplot as plt

# Perfis
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

ativos = ["PETR4", "VALE3", "ITUB4", "WEGE3", "BBAS3",
          "HGLG11", "XPLG11", "KNRI11", "MXRF11", "VISC11"]

# Dados simulados
retornos = np.array([0.15, 0.12, 0.10, 0.18, 0.11,
                     0.08, 0.09, 0.07, 0.06, 0.085])

dividendos = np.array([0.10, 0.07, 0.08, 0.02, 0.09,
                       0.11, 0.10, 0.09, 0.12, 0.095])

riscos = np.array([0.20, 0.18, 0.12, 0.25, 0.14,
                   0.10, 0.11, 0.09, 0.08, 0.10])

NUM_ATIVOS = len(ativos)
PESO_MIN = 0.05
PESO_MAX = 0.35

#Fazer com que a carteira respeite PESO_MIN e PESO_MAX
def ajustar_carteira(pesos):
    pesos = np.clip(pesos, PESO_MIN, PESO_MAX)
    pesos = pesos / np.sum(pesos)

    for i in range(len(pesos)):
        if pesos[i] > PESO_MAX:
            pesos[i] = PESO_MAX
        if pesos[i] < PESO_MIN:
            pesos[i] = PESO_MIN

    return pesos / np.sum(pesos)

#fitness --> maximizar retorno e dividendos e minimizar risco
def fitness(carteira, perfil, penalizacao=True):
    retorno = np.sum(carteira * retornos)
    dividendos_total = np.sum(carteira * dividendos)
    risco = np.sum(carteira * riscos)

    score = (
        perfil["retorno"] * retorno +
        perfil["dividendos"] * dividendos_total -
        perfil["risco"] * risco
    )

    if penalizacao:
        concentracao = np.sum(carteira**2)
        score -= 0.2 * concentracao

    return score


#PSO
def pso(perfil, penalizacao=True):
    NUM_PARTICULAS = 50
    ITERACOES = 200

    w = 0.7      # inércia
    c1 = 1.5     # cognitivo
    c2 = 1.5     # social

    # Inicialização
    particulas = [ajustar_carteira(np.random.rand(NUM_ATIVOS))
                  for _ in range(NUM_PARTICULAS)]

    velocidades = [np.random.rand(NUM_ATIVOS) * 0.1
                   for _ in range(NUM_PARTICULAS)]

    pbest = particulas.copy()
    pbest_scores = [fitness(p, perfil, penalizacao) for p in particulas]

    gbest = pbest[np.argmax(pbest_scores)]
    gbest_score = max(pbest_scores)

    historico = []

    # Loop principal
    for _ in range(ITERACOES):
        for i in range(NUM_PARTICULAS):
            r1, r2 = random.random(), random.random()

            # Atualização da velocidade
            velocidades[i] = (
                w * velocidades[i] +
                c1 * r1 * (pbest[i] - particulas[i]) +
                c2 * r2 * (gbest - particulas[i])
            )

            # Atualização da posição
            particulas[i] = particulas[i] + velocidades[i]

            # Ajustar carteira
            particulas[i] = ajustar_carteira(particulas[i])

            # Avaliar fitness
            score = fitness(particulas[i], perfil, penalizacao)

            # Atualizar pbest
            if score > pbest_scores[i]:
                pbest[i] = particulas[i]
                pbest_scores[i] = score

        # Atualizar gbest
        melhor_idx = np.argmax(pbest_scores)
        if pbest_scores[melhor_idx] > gbest_score:
            gbest = pbest[melhor_idx]
            gbest_score = pbest_scores[melhor_idx]

        historico.append(gbest_score)

    return gbest, historico


# =========================
# GRÁFICOS
# =========================

# Gráfico 1: Perfis
plt.figure(figsize=(8,5))

for nome, perfil in perfis.items():
    _, hist = pso(perfil, penalizacao=True)
    plt.plot(hist, label=nome)

plt.title("PSO - Comparação entre Perfis")
plt.xlabel("Iteração")
plt.ylabel("Fitness")
plt.legend()
plt.savefig("pso_perfis.png", dpi=300)
plt.show()


# Gráfico 2: Penalização
perfil = perfis["dividendos"]

_, hist_sem = pso(perfil, penalizacao=False)
_, hist_com = pso(perfil, penalizacao=True)

plt.figure(figsize=(8,5))
plt.plot(hist_sem, label="Sem Penalização")
plt.plot(hist_com, label="Com Penalização")

plt.title("PSO - Impacto da Penalização")
plt.xlabel("Iteração")
plt.ylabel("Fitness")
plt.legend()
plt.savefig("pso_penalizacao.png", dpi=300)
plt.show()


# Resultado final
melhor, _ = pso(perfil, penalizacao=True)

print("\nMelhor carteira (PSO):")
for a, p in zip(ativos, melhor):
    print(a, f"{p:.2%}")