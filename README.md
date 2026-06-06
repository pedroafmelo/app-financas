# App Finanças Avançadas

**Autor:** Pedro Melo  
**Disciplina:** Tópicos Avançados em Finanças — UFPB  
**Orientadores:** Dr. Orleans Martins, Dr. Felipe Pontes e Ms. Thatiane Oliveira

---

## Visão Geral

Dashboard interativo em Streamlit voltado para análise de investimentos no mercado brasileiro de ações, desenvolvido como projeto da disciplina de Tópicos Avançados em Finanças da UFPB. O app integra três grandes eixos temáticos da disciplina: **CAPM e modelos de fatores de risco**, **econometria de séries de retornos** e **seleção e simulação de carteiras**.

A metodologia de fatores segue o padrão do [NEFIN (USP)](https://nefin.com.br/), calculando seis fatores de risco para o mercado brasileiro, e os dados de preços e volumes são extraídos da plataforma [Economática](https://www.economatica.com/), enquanto a taxa livre de risco (CDI) é obtida via SGS do Banco Central do Brasil.

---

## Funcionalidades

### Screening de Ativos
Filtra ações do universo elegível com base em exposição a fatores de risco estimada por regressão:

- Suporta quatro especificações de modelo: **CAPM**, **3 fatores** (Mkt-Rf, SMB, HML), **5 fatores** (+WML, IML) e **modelo completo com 6 fatores** (+BAB)
- Exibe betas estimados via janela móvel de 24 meses para cada fator selecionado
- Filtra por setor, subsetor e classe do ativo

### Simulador de Retornos
Modelagem econométrica de retornos logarítmicos de ativos ou carteiras ponderadas:

- Modelos de volatilidade condicional: **ARCH**, **GARCH**, **EGARCH**, **GJR-GARCH**
- Especificações de média: Constante, Zero, Mínimos Quadrados
- Distribuições de erros: Normal, Gaussiana, T, T-Student
- Previsão de volatilidade e retorno por simulação, analítico ou bootstrap
- Seleção automática de melhor modelo via grid search paralelizado (MAPE / RMSE)

---

## Fatores de Risco Calculados

| Fator | Descrição |
|-------|-----------|
| **Mkt-Rf** | Prêmio de risco de mercado |
| **SMB** | Tamanho (Small Minus Big) |
| **HML** | Valor (High Minus Low book-to-market) |
| **WML** | Momentum (Winners Minus Losers) |
| **IML** | Liquidez (Illiquid Minus Liquid) |
| **BAB** | Anomalia de beta (Betting Against Beta) |

A elegibilidade dos ativos segue a metodologia NEFIN: 80% dos pregões negociados no ano anterior, volume médio diário acima de R$ 500 mil e listagem anterior a dezembro do ano anterior.

---

## Instalação e Execução

**Pré-requisito:** Python 3.12+

```bash
# Instalar dependências (recomendado)
uv sync

# Alternativa com pip
pip install -r requirements.txt

# Executar o app
cd src/app
streamlit run app.py
```

O app estará disponível em `http://localhost:8501`.

---

## Fontes de Dados

| Dado | Fonte |
|------|-------|
| Preços e volumes de ações | [Economática](https://www.economatica.com/) |
| Taxa livre de risco (CDI) | [SGS — Banco Central do Brasil](https://www.bcb.gov.br/estatisticas/tabelaespecial) |
| Dados cadastrais das empresas | Economática |

Os arquivos de dados ficam no diretório `data/` (não versionado) e incluem séries históricas de cotações, volumes, fatores consolidados e composição dos portfólios por ano.

---

## Estrutura do Projeto

```
src/app/
├── app.py                  # Entrada do app multi-página
├── screening_ativos.py     # Página de screening por fatores
├── simulacao_retornos.py   # Página do simulador econométrico
├── calculating_factors.py  # Classe RiskFactors — pipeline de cálculo dos fatores
├── backtest.py             # Classe Backtest — betas e retornos esperados
├── modelagem_retornos.py   # Classe EconometricPredict — modelos ARCH/GARCH
├── graphs.py               # Visualizações ECharts
├── utils.py                # Utilitários (config, sanitização de dados)
└── config.yaml             # Configuração central (caminhos, textos)
```

---

## Tecnologias

- [Python 3.12](https://docs.python.org/3/)
- [Streamlit](https://docs.streamlit.io/)
- [arch](https://arch.readthedocs.io/) — modelos ARCH/GARCH
- [statsmodels](https://www.statsmodels.org/) — econometria
- [Apache ECharts](https://echarts.apache.org/) via streamlit-echarts
- [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/)

---

## Referências

- [Metodologia NEFIN](https://nefin.com.br/resources/NEFIN_methodology.pdf)
- [Fama & French — CAPM: Theory and Evidence](https://mba.tuck.dartmouth.edu/bespeneckbo/default/AFA611-Eckbo%20web%20site/AFA611-S6B-FamaFrench-CAPM-JEP04.pdf)
- [Frazzini & Pedersen — Betting Against Beta](https://doi.org/10.1016/j.jfineco.2013.10.005)
- [5 Fatores Aplicados ao Brasil](https://lume.ufrgs.br/handle/10183/294754)
