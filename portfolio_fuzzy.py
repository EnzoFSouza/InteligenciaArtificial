import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Funções de pertinência (triangulares)
# -----------------------------
def triangular(x, a, b, c):
    if x <= a or x >= c:
        return 0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x < c:
        return (c - x) / (c - b)
    else:
        return 0

# -----------------------------
# Funções de pertinência (trapezoidal)
# -----------------------------
def trapezoidal(x, a, b, c, d):
    if x <= a or x >= d:
        return 0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1
    elif c < x < d:
        return (d - x) / (d - c)
    else:
        return 0
    
# -----------------------------
# Fuzzificação
# -----------------------------
def fuzzificar(valorizacao, risco, dividendos):
    fuzz = {}

    # Valorização
    fuzz['val_baixa'] = trapezoidal(valorizacao, 0, 0, 2, 5)
    fuzz['val_media'] = triangular(valorizacao, 2, 5, 8)
    fuzz['val_alta']  = trapezoidal(valorizacao, 5, 8, 10, 10)

    # Risco
    fuzz['risco_baixo'] = trapezoidal(risco, 0, 0, 2, 5)
    fuzz['risco_medio'] = triangular(risco, 2, 5, 8)
    fuzz['risco_alto']  = trapezoidal(risco, 5, 8, 10, 10)

    # Dividendos
    fuzz['div_baixo'] = trapezoidal(dividendos, 0, 0, 2, 5)
    fuzz['div_medio'] = triangular(dividendos, 2, 5, 8)
    fuzz['div_alto']  = trapezoidal(dividendos, 5, 8, 10, 10)

    return fuzz


# -----------------------------
# Regras fuzzy (Mamdani)
# -----------------------------
def regras(fuzz):
    regras = {}

    # Regra 1: Se valorização alta E risco baixo → boa
    regras['boa1'] = min(fuzz['val_alta'], fuzz['risco_baixo'])

    # Regra 2: Se dividendos altos E risco baixo → boa
    regras['boa2'] = min(fuzz['div_alto'], fuzz['risco_baixo'])

    # Regra 3: Se valorização média → média
    regras['media1'] = fuzz['val_media']

    # Regra 4: Se dividendos médios → média
    regras['media2'] = fuzz['div_medio']

    # Regra 5: Se risco alto → ruim
    regras['ruim1'] = fuzz['risco_alto']

    # Regra 6: Se valorização baixa → ruim
    regras['ruim2'] = fuzz['val_baixa']

    return regras


# -----------------------------
# Agregação (Mamdani → max)
# -----------------------------
def agregacao(regras):
    grau_ruim = max(regras['ruim1'], regras['ruim2'])
    grau_media = max(regras['media1'], regras['media2'])
    grau_boa = max(regras['boa1'], regras['boa2'])

    return grau_ruim, grau_media, grau_boa


# -----------------------------
# Defuzzificação (centroide)
# -----------------------------
def defuzzificacao(grau_ruim, grau_media, grau_boa):
    x = np.linspace(0, 10, 100)

    ruim = [min(grau_ruim, triangular(i, 0, 0, 5)) for i in x]
    media = [min(grau_media, triangular(i, 2, 5, 8)) for i in x]
    boa = [min(grau_boa, triangular(i, 5, 10, 10)) for i in x]

    agregado = np.maximum(ruim, np.maximum(media, boa))

    numerador = np.sum(x * agregado)
    denominador = np.sum(agregado)

    if denominador == 0:
        return 0

    return numerador / denominador


# -----------------------------
# Classificação final
# -----------------------------
def classificar(valor):
    if valor < 4:
        return "Ruim"
    elif valor < 7:
        return "Média"
    else:
        return "Boa"


def plot_defuzzificacao(grau_ruim, grau_media, grau_boa):

    x = np.linspace(0, 10, 200)

    # Funções de saída
    ruim = np.array([min(grau_ruim, trapezoidal(i, 0, 0, 2, 5)) for i in x])
    media = np.array([min(grau_media, triangular(i, 2, 5, 8)) for i in x])
    boa = np.array([min(grau_boa, trapezoidal(i, 5, 8, 10, 10)) for i in x])

    # Agregação (máximo)
    agregado = np.maximum(ruim, np.maximum(media, boa))

    # Centroide
    numerador = np.sum(x * agregado)
    denominador = np.sum(agregado)

    centroide = numerador / denominador if denominador != 0 else 0

    # Plot
    plt.figure(figsize=(8,5))

    # Curvas individuais
    plt.plot(x, ruim, linestyle='--', label="Ruim")
    plt.plot(x, media, linestyle='--', label="Média")
    plt.plot(x, boa, linestyle='--', label="Boa")

    # Área agregada
    plt.fill_between(x, agregado, alpha=0.3)

    # Linha do centroide
    plt.axvline(centroide, linestyle='-', label=f"Centroide = {centroide:.2f}")

    plt.title("Defuzzificação (Método do Centroide)")
    plt.xlabel("Score")
    plt.ylabel("Pertinência")

    plt.legend()
    plt.grid()

    plt.savefig(f"{PATH}defuzzificacao.png", dpi=300)
    plt.show()

    return centroide

# -----------------------------
# Função principal
# -----------------------------
def classificador_fuzzy(valorizacao, risco, dividendos):
    fuzz = fuzzificar(valorizacao, risco, dividendos)
    regras_ativas = regras(fuzz)
    grau_ruim, grau_media, grau_boa = agregacao(regras_ativas)
    valor_final = plot_defuzzificacao(grau_ruim, grau_media, grau_boa)
    #valor_final = defuzzificacao(grau_ruim, grau_media, grau_boa)
    classe = classificar(valor_final)

    return valor_final, classe


# -----------------------------
# Teste
# -----------------------------
PATH = "graficos/fuzzy/"
valorizacao = 7
risco = 3
dividendos = 6

valor, classe = classificador_fuzzy(valorizacao, risco, dividendos)

print(f"Score final: {valor:.2f}")
print(f"Classificação: {classe}")

x = np.linspace(0, 10, 100)

baixa = [trapezoidal(i, 0, 0, 2, 5) for i in x]
media = [triangular(i, 2, 5, 8) for i in x]
alta  = [trapezoidal(i, 5, 8, 10, 10) for i in x]

plt.figure(figsize=(8,5))

plt.plot(x, baixa, label="Baixa")
plt.plot(x, media, label="Média")
plt.plot(x, alta, label="Alta")

plt.title("Funções de Pertinência (Fuzzy)")
plt.xlabel("Valor")
plt.ylabel("Pertinência")

plt.legend()
plt.grid()

plt.savefig(f"{PATH}fuzzy_trap_tri.png", dpi=300)
plt.show()