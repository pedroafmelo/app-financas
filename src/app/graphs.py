# -*- coding: UTF-8 -*-
""" Import Modules """
import streamlit as st
from datetime import datetime, date, timedelta
from os import path
from pathlib import Path
import pandas as pd
import numpy as np

# local imports
from utils import get_config, update_config



def correlation_heatmap(corr):
    columns = list(corr.columns)
    data = []

    for i in range(len(columns)):
        for j in range(len(columns)):
            if i > j: 
                valor = round(float(corr.iloc[i, j]), 2)
                data.append([j, i, valor])

    option = {
        "backgroundColor": "#0E1117",
        "title": {
            "text": "Correlation Heatmap - Fatores",
            "textStyle": {"color": "#FFFFFF"}
        },
        "grid": {
            "top": 70,
            "bottom": 140,
            "left": 70,
            "right": 80
        },
        "xAxis": {
                    "type": "category",
                    "data": columns[:-1], 
                    "axisLabel": {"color": "#FFFFFF", "rotate": 45, "interval": 0},
                    "splitLine": {"show": False},
                    "axisTick": {"show": False}
                },
        "yAxis": {
                    "type": "category",
                    "data": columns[1:], 
                    "axisLabel": {"color": "#FFFFFF"},
                    "splitLine": {"show": False},
                    "axisTick": {"show": False}
                },
        "visualMap": {
            "min": -1,
            "max": 1,
            "orient": "horizontal",
            "calculable": True,
            "right": 550,
            "inRange": {
                "color": ["#8b0000", "#ffffff", "#006400"]
            },
            "textStyle": {
                "color": "#FFFFFF"
            },
        },
        "series": [
            {
                "name": "Correlação",
                "type": "heatmap",
                "data": [[d[0], d[1]-1, d[2]] for d in data],
                "label": {
                    "show": True,
                    "color": "#000000",
                    "fontSize": 10
                },
                "itemStyle": {
                    "emphasis": {
                        "shadowBlur": 10,
                        "shadowColor": "rgba(0, 0, 0, 0.5)"
                    }
                }
            }
        ]
    }

    return option

def histogram(df, var):

    values = df[var]

    if values.nunique() <= 10 or values.dtype == "object":
        freq = values.value_counts().sort_index()
        x_labels = freq.index.tolist()
        data = freq.tolist()
    else:
        counts, bin_edges = np.histogram(values, bins=200)
        x_labels = [f"{(bin_edges[i] * 100):.2f} - {(bin_edges[i+1] * 100):.2f}" for i in range(len(bin_edges)-1)]
        data = counts.tolist()

    option = {
        "color": "#fba725",
        "tooltip": {"trigger": "axis", 
                    "axisPointer": {
                        "type": "shadow"
                    },},
        "backgroundColor": "#0E1117",
        "toolbox": {
            "feature": {
                "dataZoom": {"yAxisIndex": 'none'},
                "restore": {},
                "saveAsImage": {}
            }
        },
        "dataZoom": [
            {"show": True, "realtime": True, "start": 0, "end": 100},
            {"type": 'inside', "realtime": True, "start": 0, "end": 100}
        ],
        "grid": {
            "left": "0%", 
            "right": "5%", 
            "containLabel": True
        },
        "title": {
            "text": f"Histograma - {var.replace("_", " ").title()}",
            "textStyle": {"color": "#FFFFFF"}
        },
        "xAxis": {
            "type": "category",
            "data": x_labels,
            "axisLabel": {"rotate": 45}
        },
        "yAxis": {"type": "value"},
        "series": [{
            "data": data,
            "type": "bar",
            
        }]
    }

    return option


def heatmap(time_cuts, factors, dataset):

    data = []

    dataset = dataset.set_index("Fator")

    y_labels = dataset.index.tolist()
    x_labels = dataset.columns.tolist()

    for y_idx, fator in enumerate(y_labels):
        for x_idx, coluna in enumerate(x_labels):
            valor = dataset.loc[fator, coluna]
            data.append([x_idx, y_idx, float(valor)])


    option = {
        "tooltip": {"trigger": "axis", 
                    "axisPointer": {
                        "type": "shadow"
                    },},
        "backgroundColor": "#0E1117",
        "toolbox": {
            "feature": {
                "dataZoom": {"yAxisIndex": 'none'},
                "restore": {},
                "saveAsImage": {}
            }
        },
        "grid": {
            "top": 70,
            "bottom": 90,
            "left": 70,
            "right": 80
        },
        "title": {
            "text": "Retornos dos fatores por corte temporal",
            "textStyle": {"color": "#FFFFFF"}
        },
        "tooltip": {
            "position": 'top'
        },
        "xAxis": {
            "type": 'category',
            "data": time_cuts,
            "splitArea": {
            "show": True
            },
            "axisLabel": {
                        "color": "#FFFFFF"
                    }
        },
        "yAxis": {
            "type": 'category',
            "data": factors,
            "splitArea": {
                "show": True
            },
        "axisLabel": {
                        "color": "#FFFFFF"
                    }
        },
        "visualMap": {
            "min": -30,
            "max": 40,
            "orient": "horizontal",
            "calculable": True,
            "right": 550,
            "inRange": {
                "color": ["#8b0000", "#ffffff", "#006400"]
            },
            "textStyle": {
                "color": "#FFFFFF"
            },
        },
        "series": [
            {
            "name": 'Retorno anualizado',
            "type": 'heatmap',
            "data": data,
            "label": {
                "show": True
            },
            "label": {
                    "show": True,
                    "formatter": "{@[2]} %",
                },
                "itemStyle": {
                    "borderRadius": [6, 6, 6, 6]
                }
            }
        ]
        }

    return option


def returns_forecast_chart(hist_series: pd.Series, pred_ret: pd.DataFrame, title: str = "") -> dict:
    """
    hist_series : Series com índice datetime e retornos em % (log-retornos * 100).
    pred_ret    : forecasts.mean do arch — último row contém h.01..h.N horizontes.
    """
    hist_dates = [
        d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
        for d in hist_series.index
    ]
    hist_values = [round(float(v), 4) for v in hist_series.values]

    forecast_row = pred_ret.iloc[-1]
    last_date = hist_series.index[-1]
    if hasattr(last_date, "to_pydatetime"):
        last_date = last_date.to_pydatetime()

    horizon = len(forecast_row)
    future_dates, future_values = [], []
    for i, val in enumerate(forecast_row.values):
        future_dates.append((last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d"))
        future_values.append(round(float(val), 4))

    hist_n = len(hist_values)
    all_dates = hist_dates + future_dates

    return {
        "backgroundColor": "#0E1117",
        "title": {
            "text": f"Retornos Históricos e Previstos{' — ' + title if title else ''}",
            "textStyle": {"color": "#FFFFFF", "fontSize": 14},
        },
        "tooltip": {"trigger": "axis"},
        "legend": {
            "data": ["Retornos Históricos", "Retornos Previstos"],
            "textStyle": {"color": "#FFFFFF"},
            "bottom": 0,
        },
        "toolbox": {
            "feature": {
                "dataZoom": {"yAxisIndex": "none"},
                "restore": {},
                "saveAsImage": {},
            }
        },
        "dataZoom": [
            {"show": True, "realtime": True, "start": 98, "end": 100},
            {"type": "inside", "realtime": True},
        ],
        "grid": {"left": "0%", "right": "5%", "containLabel": True},
        "xAxis": {
            "type": "category",
            "data": all_dates,
            "axisLabel": {"color": "#AAAAAA", "rotate": 45},
        },
        "yAxis": {
            "type": "value",
            "axisLabel": {"color": "#AAAAAA", "formatter": "{value}%"},
            "splitLine": {"lineStyle": {"color": "#333"}},
        },
        "series": [
            {
                "name": "Retornos Históricos",
                "type": "line",
                "data": hist_values + [None] * horizon,
                "lineStyle": {"color": "#f5770c", "width": 1.5},
                "symbol": "none",
                "areaStyle": {"color": "rgba(245,119,12,0.06)"},
                "connectNulls": False,
            },
            {
                "name": "Retornos Previstos",
                "type": "line",
                "data": [None] * hist_n + future_values,
                "lineStyle": {"color": "#4CAF50", "width": 2, "type": "dashed"},
                "itemStyle": {"color": "#4CAF50"},
                "symbol": "circle",
                "symbolSize": 7,
                "connectNulls": False,
            },
        ],
    }


def volatility_forecast_chart(hist_series: pd.Series, pred_var: pd.DataFrame, title: str = "") -> dict:
    """
    hist_series : Series com índice datetime e retornos em % — usada para calcular vol histórica.
    pred_var    : forecasts.variance do arch — sqrt converte para desvio padrão (vol prevista).
    """
    rolling_vol = hist_series.rolling(window=21).std().dropna()
    hist_dates = [
        d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
        for d in rolling_vol.index
    ]
    hist_vol_vals = [round(float(v), 4) for v in rolling_vol.values]

    forecast_row = pred_var.iloc[-1]
    last_date = hist_series.index[-1]
    if hasattr(last_date, "to_pydatetime"):
        last_date = last_date.to_pydatetime()

    horizon = len(forecast_row)
    future_dates, future_vol_vals = [], []
    for i, val in enumerate(forecast_row.values):
        future_dates.append((last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d"))
        future_vol_vals.append(round(float(np.sqrt(abs(val))), 4))

    hist_n = len(hist_vol_vals)
    all_dates = hist_dates + future_dates

    return {
        "backgroundColor": "#0E1117",
        "title": {
            "text": f"Volatilidade Histórica e Prevista{' — ' + title if title else ''}",
            "textStyle": {"color": "#FFFFFF", "fontSize": 14},
        },
        "tooltip": {"trigger": "axis"},
        "legend": {
            "data": ["Volatilidade Histórica (21d)", "Volatilidade Prevista (GARCH)"],
            "textStyle": {"color": "#FFFFFF"},
            "bottom": 0,
        },
        "toolbox": {
            "feature": {
                "dataZoom": {"yAxisIndex": "none"},
                "restore": {},
                "saveAsImage": {},
            }
        },
        "dataZoom": [
            {"show": True, "realtime": True, "start": 98, "end": 100},
            {"type": "inside", "realtime": True},
        ],
        "grid": {"left": "0%", "right": "5%", "containLabel": True},
        "xAxis": {
            "type": "category",
            "data": all_dates,
            "axisLabel": {"color": "#AAAAAA", "rotate": 45},
        },
        "yAxis": {
            "type": "value",
            "axisLabel": {"color": "#AAAAAA", "formatter": "{value}%"},
            "splitLine": {"lineStyle": {"color": "#333"}},
        },
        "series": [
            {
                "name": "Volatilidade Histórica (21d)",
                "type": "line",
                "data": hist_vol_vals + [None] * horizon,
                "lineStyle": {"color": "#f5770c", "width": 1.5},
                "symbol": "none",
                "areaStyle": {"color": "rgba(245,119,12,0.06)"},
                "connectNulls": False,
            },
            {
                "name": "Volatilidade Prevista (GARCH)",
                "type": "line",
                "data": [None] * hist_n + future_vol_vals,
                "lineStyle": {"color": "#4169E1", "width": 2, "type": "dashed"},
                "itemStyle": {"color": "#4169E1"},
                "symbol": "circle",
                "symbolSize": 7,
                "connectNulls": False,
            },
        ],
    }