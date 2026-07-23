import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

plt.style.use('dark_background')

df = pd.read_csv('dataset.csv')

colunas_remover = ['Person ID', 'Gender', 'Occupation', 'Blood Pressure', 'Daily Steps', 'Sleep Disorder']
df = df.drop(columns=[col for col in colunas_remover if col in df.columns])

print(70*"-")
print(df.head())
print(70*"-")

df_prep = pd.get_dummies(df)

plt.figure(figsize=(12, 8))
correlacao = df_prep.corr()
sns.heatmap(correlacao, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Mapa de Correlação (Incluindo Categorias de IMC)")
plt.show()

target_col = 'Stress Level' if 'Stress Level' in df_prep.columns else 'stress_level'

condicoes = [
    (df_prep[target_col] <= 4),
    (df_prep[target_col] >= 5) & (df_prep[target_col] <= 7),
    (df_prep[target_col] >= 8)
]
valores = [0, 1, 2]

y = np.select(condicoes, valores)
X = df_prep.drop(columns=[target_col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modelos = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier()
}

resultados = []
matrizes = {}

print("Treinando os modelos no conjunto de teste local...")

for nome, modelo in modelos.items():
    modelo.fit(X_train_scaled, y_train)
    y_pred = modelo.predict(X_test_scaled)

    resultados.append({
        "Modelo": nome,
        "Acurácia": accuracy_score(y_test, y_pred),
        "Precisão": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    })

    matrizes[nome] = np.round(confusion_matrix(y_test, y_pred, normalize='true') * 100, 2)

df_resultados = pd.DataFrame(resultados)
print("\nRESULTADOS FINAIS (CONJUNTO DE TESTE)")
print(70*"-")
print(df_resultados)
print(70*"-")

nomes_categorias = ['Baixo', 'Médio', 'Alto']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (nome, matriz) in zip(axes, matrizes.items()):
    sns.heatmap(matriz, annot=True, fmt='.0f', cmap='Blues', cbar=False, ax=ax,
                xticklabels=nomes_categorias, yticklabels=nomes_categorias)

    for text in ax.texts:
        text.set_text(text.get_text() + '%')

    ax.set_title(nome, fontsize=14, pad=10)
    ax.set_xlabel("Estresse Previsto pela IA", fontsize=11)
    ax.set_ylabel("Estresse Real do Paciente", fontsize=11)

plt.tight_layout()
plt.show()

print("Calculando Validação Cruzada Múltipla...")

scoring = {
    'Acurácia': 'accuracy',
    'Precisão': 'precision_weighted',
    'Recall': 'recall_weighted',
    'F1-Score': 'f1_weighted'
}

resultados_cv = {}

for nome, modelo in modelos.items():
    scores = cross_validate(modelo, X_train_scaled, y_train, cv=5, scoring=scoring)

    resultados_cv[nome] = {
        'Acurácia': (np.mean(scores['test_Acurácia']), np.std(scores['test_Acurácia'])),
        'Precisão': (np.mean(scores['test_Precisão']), np.std(scores['test_Precisão'])),
        'Recall': (np.mean(scores['test_Recall']), np.std(scores['test_Recall'])),
        'F1-Score': (np.mean(scores['test_F1-Score']), np.std(scores['test_F1-Score']))
    }

metricas = list(scoring.keys())
nomes_modelos = list(modelos.keys())

x = np.arange(len(nomes_modelos))
largura_barra = 0.2

fig, ax = plt.subplots(figsize=(12, 7))
cores = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']

for i, metrica in enumerate(metricas):
    medias = [resultados_cv[m][metrica][0] for m in nomes_modelos]
    desvios = [resultados_cv[m][metrica][1] for m in nomes_modelos]

    posicao = x - (largura_barra * 1.5) + (i * largura_barra)

    ax.bar(posicao, medias, largura_barra, yerr=desvios, label=metrica,
           capsize=6, color=cores[i], alpha=0.85, edgecolor='black')

ax.set_ylabel('Pontuação Média (0 a 1.0)', fontsize=12)
ax.set_title('Desempenho dos Modelos na Validação Cruzada (Métricas c/ Desvio Padrão)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(nomes_modelos, fontsize=12)
ax.set_ylim(0, 1.1)
ax.legend(loc='lower right', fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()