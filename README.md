# Desafio Cientista de Dados – IMDb (LH_CD_Guilherme_Silveira_Araujo)

Solução do desafio da Indicium: **EDA, respostas de negócio e um modelo preditivo** para estimar a nota do IMDb (0–10) a partir de características de filmes.

---

## 0) Visão Geral

**Problema:** prever `IMDB_Rating` (variável contínua) ⇒ regressão supervisionada.

### Pipeline de Features
- **Numéricas lineares:** `Released_Year`, `Runtime` → `StandardScaler`
- **Numéricas assimétricas:** `Gross`, `budget`, `No_of_Votes` → `log1p` + `StandardScaler`
- **Categórica baixa cardinalidade:** `Certificate` → `OneHotEncoder(handle_unknown="ignore")`
- **Categóricas alta cardinalidade:** `Director`, `Star1`, `Star2`, `Star3` → *Frequency Encoding* (categorias novas → 0.0)
- **Gêneros (Action … Western):** variáveis binárias (passthrough)

### Modelo Vencedor
**CatBoost Regressor**

### Métrica Principal
**RMSE** (penaliza mais erros grandes).  
MAE e R² foram usadas como métricas complementares.

### Desempenho no Conjunto de Teste

| Modelo               | MAE   | RMSE  | R²    |
|----------------------|-------|-------|-------|
| Regressão Linear     | 0.520 | 0.730 | 0.594 |
| Random Forest        | 0.463 | 0.664 | 0.663 |
| HistGradientBoosting | 0.457 | 0.650 | 0.678 |
| XGBoost              | 0.449 | 0.644 | 0.684 |
| **CatBoost**         | **0.444** | **0.643** | **0.685** |

**Previsão do enunciado:** *The Shawshank Redemption*  
Previsto: **8.93** vs. real: **9.30** → erro absoluto **0.37 (~3,98%)**, menor que o MAE do modelo.

---

## 1) Estrutura do Repositório
```text
├─ README.md
├─ requirements.txt
├─ models/
│ └─ imdb_catboost_pipeline.pkl
├─ notebooks/
│ ├─ 01_EDA.ipynb
│ ├─ 01_2_extra_data.ipynb
│ ├─ 02_perguntas_negocio.ipynb
│ └─ 03_04_05_IMDB_PREVISAO.ipynb
├─ reports/
│ ├─ 01_eda.pdf
│ ├─ 01_2_extra_data.pdf
│ └─ 02_perguntas_negocio.pdf
│ └─ 03_04_05_IMDB_PREVISAO.pdf
├─ data/
│ ├─ raw/
│ └─ processed/
│ └─ df_eda01.csv
└─ .gitignore
```
### Mapa dos Entregáveis

- **Tópico 1 — Análise Exploratória e Hipóteses**  
  As hipóteses levantadas e a análise exploratória requisitada estão no notebook  
  `notebooks/01_eda.ipynb` e no relatório correspondente em `reports/01_eda.pdf`.

- **Tópico 2 — Perguntas de Negócio**  
  As respostas às perguntas solicitadas estão documentadas no notebook  
  `notebooks/02_perguntas_negocio.ipynb` e no relatório em `reports/02_perguntas_negocio.pdf`.

- **Tópicos 3, 4 e 5 — Modelagem, Previsão e Salvamento do Modelo**  
  Todo o processo de pré-processamento, comparação de modelos, métricas, previsão do enunciado e salvamento do pipeline em `.pkl` está no notebook  
  `notebooks/03_04_05_IMDB_PREVISAO.ipynb` e no relatório em `reports/03_04_05_IMDB_PREVISAO.pdf`.


## 2) Instalação

O projeto utiliza **Python 3.10+**.  
Recomenda-se criar um ambiente virtual antes de instalar as dependências.

### Passos:

1. Clone este repositório:
   ```bash
   git clone https://github.com/SEU_USUARIO/LH_CD_Guilherme_Silveira_Araujo.git
   cd LH_CD_Guilherme_Silveira_Araujo
    ```
1.1 Caso opte por baixar diretamente a pasta:
        Abra o terminal (Prompt de Comando no Windows, ou Terminal no macOS/Linux) e navegue até a pasta:

        cd caminho/para/LH_CD_Guilherme_Silveira_Araujo

2. Crie e ative o ambiente virtual:
     ```bash
    python -m venv .venv
     ```
    **Windows**
    ```bash
    .venv\Scripts\activate
     ```
    **macOS/Linux**
    ```bash
    source .venv/bin/activate
     ```

4. Instale as dependências:
     ```bash
    pip install -r requirements.txt
     ```

5. Abra os notebooks para rodar os experimentos e análises!!


## 3) Execução

Os notebooks devem ser executados em ordem:

1. 01_eda.ipynb – análise exploratória e hipóteses
2. 01_2_extra_data.ipynb – dados extras incorporados à EDA
3. 02_perguntas_negocio.ipynb – respostas ao tópico 2 do enunciado
4. 03_04_05_IMDB_PREVISAO.ipynb – pré-processamento, modelagem, previsão e salvamento do modelo .pkl

## 4) Como usar o modelo(.pkl)

O arquivo models/imdb_catboost_pipeline.pkl contém:

- preprocessor (ColumnTransformer já fitado com todas as transformações),
- model (CatBoost treinado),
- feature_cols (lista de colunas brutas esperadas),
- metadata (versões do ambiente).

### Exemplo de Previsão

```python
from pathlib import Path
import joblib
import pandas as pd

# Carregar pipeline completo
art = joblib.load(Path("models/imdb_catboost_pipeline.pkl"))
preprocessor = art["preprocessor"]
model        = art["model"]
feature_cols = art["feature_cols"]

# Exemplo de novo filme
novo_filme = {
    "Series_Title": "The Shawshank Redemption",
    "Released_Year": "1994",
    "Certificate": "A",
    "Runtime": "142 min",
    "Genre": "Drama",
    "Overview": "Two imprisoned men bond over a number of years...",
    "Meta_score": 80.0,
    "Director": "Frank Darabont",
    "Star1": "Tim Robbins",
    "Star2": "Morgan Freeman",
    "Star3": "Bob Gunton",
    "Star4": "William Sadler",
    "No_of_Votes": 2343110,
    "Gross": "28,341,469"
}
df_novo = pd.DataFrame([novo_filme])

# Transformar e prever
X_novo = preprocessor.transform(df_novo[feature_cols])
pred = model.predict(X_novo)
print("Nota prevista:", round(float(pred[0]), 2))

```

**OBS:**
Importante: não refaça .fit() no preprocessor para novos dados — use sempre .transform().

## 5) Salvamento e Carregamento do Modelo


```python
from pathlib import Path
import joblib, sklearn, sys

artifact = {
  "preprocessor": preprocessor,
  "model": cat,  # CatBoost treinado
  "feature_cols": num_linear + num_skewed + cat_low + cat_high + genre_cols,
  "metadata": {"python": sys.version, "sklearn": sklearn.__version__}
}

Path("models").mkdir(exist_ok=True)
joblib.dump(artifact, Path("models") / "imdb_catboost_pipeline.pkl")
```

Carregar novamente:

```python
import joblib
art = joblib.load("models/imdb_catboost_pipeline.pkl")
preprocessor = art["preprocessor"]
model        = art["model"]
feature_cols = art["feature_cols"]
```
## 6) Relatórios

Os relatórios estão na pasta reports/ e são exportações diretas dos notebooks.
Eles servem apenas como visualização complementar.
Em caso de divergência, os notebooks são a fonte de verdade.

## 7) Decisões e Justificativas (resumo)

- Tipo de problema: regressão (alvo contínuo).

- Transformações: padronização; log1p para reduzir assimetria; OneHotEncoder  para baixa cardinalidade; Frequency Encoding para alta cardinalidade; gêneros binários em passthrough.

- Modelo escolhido: CatBoost.

- Prós: excelente em dados tabulares; lida muito bem com categóricas; robusto; melhor desempenho nas métricas.

- Contras: menos interpretável que modelos lineares; custo de treino maior que o baseline.

- Métricas: RMSE (principal, penaliza erros grandes), MAE (erro médio) e R² (proporção explicada).

## 8) Reprodutibilidade

Versões dos pacotes em requirements.txt.

O .pkl inclui preprocessor, model e feature_cols.

random_state fixado nos modelos.

