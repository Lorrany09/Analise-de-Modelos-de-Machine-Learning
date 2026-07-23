# Predição de Níveis de Estresse com Machine Learning
Este projeto realiza uma análise comparativa de modelos de Aprendizado de Máquina na predição de níveis de estresse com base em indicadores de sono, saúde e estilo de vida. Utilizando a base de dados pública [Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset?resource=download), avaliando o desempenho dos algoritmos K-Nearest Neighbors (KNN), Support Vector Machines (SVM) e Árvore de Decisão (Decision Tree).

---
## Autores
Projetado e desenvolvido por:

<table align="center">
  <tr>
    <td align="center" width="25%">
      <a href="https://github.com/AndressaVilin">
        <img src="https://github.com/AndressaVilin.png" width="100px;" alt="Andressa Evilin"/><br />
        <sub><b>Andressa Evilin</b></sub>
      </a><br /><br />
      <a href="https://github.com/AndressaVilin">
        <img src="https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white" />
      </a>
    </td>
    <td align="center" width="25%">
      <a href="https://github.com/icarol347">
        <img src="https://github.com/icarol347.png" width="100px;" alt="Icaro Lamartine"/><br />
        <sub><b>Icaro Lamartine</b></sub>
      </a><br /><br />
      <a href="https://github.com/icarol347">
        <img src="https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white" />
      </a>
    </td>
    <td align="center" width="25%">
      <a href="https://github.com/gabrieloliverdev07">
        <img src="https://github.com/gabrieloliverdev07.png" width="100px;" alt="João Gabriel"/><br />
        <sub><b>João Gabriel</b></sub>
      </a><br /><br />
      <a href="https://github.com/gabrieloliverdev07">
        <img src="https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white" />
      </a>
    </td>
    <td align="center" width="25%">
      <a href="https://github.com/Lorrany09">
        <img src="https://github.com/Lorrany09.png" width="100px;" alt="Lorrany Queiroz"/><br />
        <sub><b>Lorrany Queiroz</b></sub>
      </a><br /><br />
      <a href="https://github.com/Lorrany09">
        <img src="https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white" />
      </a>
    </td>
  </tr>
</table>

---

## Visão Geral

O estresse crônico e a má qualidade do sono estão fortemente interligados, impactando de forma direta a saúde física e mental. Este trabalho avalia o uso de Inteligência Artificial para mapear o estresse com base em hábitos diários e parâmetros fisiológicos, fornecendo insumos que podem apoiar sistemas de suporte à decisão clínica e triagem médica.

---

## Tecnologias Utilizadas

- **Linguagem:** Python 3.12
- **Manipulação de Dados:** `pandas`, `numpy`
- **Visualização de Dados:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`
  - Preprocessamento: `StandardScaler`
  - Seleção de Modelo: `train_test_split`, `cross_validate`
  - Classificadores: `KNeighborsClassifier`, `SVC`, `DecisionTreeClassifier`
  - Métricas: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`

---

## Metodologia e Pipeline

### 1. Pré-processamento dos Dados
- **Limpeza de Atributos:** Colunas irrelevantes ou redundantes para a classificação do estresse (ex: `Person ID`, `Gender`, `Occupation`, `Blood Pressure`, `Daily Steps`, `Sleep Disorder`) são descartadas.
- **One-Hot Encoding (`pd.get_dummies`):** Variáveis categóricas, como a categoria de IMC, são transformadas em colunas binárias para que os modelos matemáticos consigam interpretá-las.
- **Agrupamento do Target (`Stress Level`):** A variável alvo é categorizada em três classes distintas:
  - **0 (Baixo):** Nível de estresse $\le 4$
  - **1 (Médio):** Nível de estresse entre 5 e 7
  - **2 (Alto):** Nível de estresse $\ge 8$
- **Análise Exploratória:** Gerado um mapa de calor utilizando a **Correlação de Pearson** para avaliar as relações entre qualidade do sono, duração do sono, frequência cardíaca e nível de estresse.

### 2. Divisão e Normalização
- **Divisão Treino/Teste:** 80% dos dados para treinamento e 20% mantidos em *teste cego* para avaliação final do modelo.
- **Padronização:** Aplicação de dimensionamento das variáveis no conjunto de treino e transformação nos dados de teste para equalizar as escalas das variáveis numéricas.

### 3. Algoritmos Avaliados
1. **K-Nearest Neighbors (KNN):** Algoritmo baseado em proximidade e distância geométrica entre pontos no espaço de atributos.
2. **Support Vector Machines (SVM):** Modelo que busca encontrar o hiperplano ótimo de separação entre as classes.
3. **Árvore de Decisão (Decision Tree):** Estrutura hierárquica construída a partir do ganho de informação para gerar regras de decisão claras.

### 4. Validação Cruzada
- Aplicação de **Validação Cruzada de 5 Dobras (5-Fold Cross-Validation)** sobre o conjunto de treino para mitigar o *overfitting* e analisar a estabilidade e a variação dos modelos.

---

## Resultados e Desempenho

### Métricas do Conjunto de Teste

Após o treinamento, os modelos foram submetidos à avaliação no conjunto de teste inédito (20% dos dados). A **Árvore de Decisão** apresentou o desempenho superior absoluto em todas as métricas:

| Modelo | Acurácia | Precisão | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree** | **98,67%** | **98,72%** | **98,67%** | **98,67%** |
| **KNN** | 96,00% | 96,13% | 96,00% | 95,96% |
| **SVM** | 86,67% | 88,98% | 86,67% | 86,80% |

### Análise das Matrizes de Confusão

- **Classe Estresse Baixo:** Todos os 3 algoritmos obtiveram **100% de acerto**.
- **Classe Estresse Médio:** Árvore de Decisão e KNN obtiveram **97% de acerto**, SVM obteve 78% de acerto.
- **Classe Estresse Alto (Crítico):** A **Árvore de Decisão** alcançou **100% de acerto**. O KNN e o SVM obtiveram 88% de acerto nessa classe.
