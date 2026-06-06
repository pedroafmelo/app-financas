# -*- coding: UTF-8 -*-
""" Import Modules """

from os import path
from pathlib import Path
import yaml
from datetime import datetime
import pandas as pd
import time
import numpy as np
from typing import Any, Literal
from bcb import sgs
from dateutil.relativedelta import relativedelta


def get_config() -> dict:
    """
    load config vars
    
    :return: return a dictionary containing yaml config vars
    :rtype: dict
    """

    config_path = path.join(path.dirname(__file__), "config.yaml")

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
            data["config_vars"]["data_path"] = path.join(
                path.dirname(path.dirname(path.dirname(__file__))), "data")
            static_path = path.join(path.dirname(__file__), "static")
            for key, value in data["config_vars"].items():
                if key != "data_path" and "data_feed" not in key:
                    data[key] = path.join(data["config_vars"]["data_path"], value)
                else:
                    data[key] = value

            for key, value in data["app_configs"].items():
                data[key] = path.join(static_path, value)

            for key, value in data["llm_configs"].items():
                data[key] = path.join(static_path, value)

    except (FileNotFoundError, yaml.YAMLError) as error:
        raise Exception(f"Error loading configuration: {error}") from error
    
    return data

def update_config(layer_1: str, layer_2: str, update_data):
    """
    Update config yaml file
    
    :param layer_1: 1st layer of yaml file
    :type layer_1: str
    :param layer_2: 2nd layer of yaml file
    :type layer_2: str
    """

    config_path = path.join(path.dirname(__file__), "config.yaml")

    try:
        
        with open(config_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        data[layer_1][layer_2] = update_data
        with open(config_path, "w") as file:
            yaml.safe_dump(data, file, allow_unicode=True,
                           sort_keys=False,
                           default_flow_style=False)

    except (FileNotFoundError, yaml.YAMLError) as error:
        raise Exception(f"Error updating yaml file: {error}") from error

def sanitize_df(dataframe: pd.DataFrame, 
                 cols_to_cast_float: list[str],
                 cols_to_cast_int: list[str],
                 col_to_cast_date: str,
                 br_date_format = False,
                 ticker_col: str="ticker",
                 date_formate_princip: str=None) -> pd.DataFrame:
    """
    Sanitizes a raw df
    
    :param dataframe: raw df
    :type dataframe: pd.DataFrame
    :return: sanitized df
    :rtype: DataFrame
    """

    try:
        cols_to_cast_float = [col for col in cols_to_cast_float if col in dataframe.columns]
        cols_to_cast_int = [col for col in cols_to_cast_int if col in dataframe.columns]
        date_format = "%d/%m/%Y" if br_date_format else "%Y-%m-%d"

        if date_formate_princip:
            date_format = date_formate_princip

        clean_df = (
            dataframe
            .assign(
                **{
                    ticker_col: lambda x: x[ticker_col].str.replace("<XBSP>", ""),
                    "empresa": lambda x: x[ticker_col].str.slice(0, 4)
                }
            )
            .assign(
                **{
                    col: (lambda x, c=col: pd.to_numeric(x[c], errors="coerce"))
                    if col == "close"
                    else (lambda x, c=col: pd.to_numeric(x[c].replace("-", "0")))
                    for col in cols_to_cast_float
                }
            )
            .assign(
                **{
                    col: (lambda x, c=col: x[c].replace("-", "0").astype(int))
                    for col in cols_to_cast_int
                }
            )
            .astype(
                {col: float for col in cols_to_cast_float}
            )
            .astype(
                {col: int for col in cols_to_cast_int}
            )
        )

        if col_to_cast_date:
            clean_df = (
                clean_df
                .assign(
                    **{col_to_cast_date: lambda x: pd.to_datetime(x[col_to_cast_date], format=date_format)}
            )
            )
    
    except Exception as error:
        raise Exception(f"Couldn't clean dataframe: {error}") from error
    
    return clean_df


def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

    yield pd.DataFrame(
        np.random.randn(5, 10),
        columns=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    )

    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)


def get_bcb(indicators: list[Literal["cdi_mensal", "cdi_diario"]], 
            start: datetime|str, end: datetime|str) -> pd.DataFrame:
    
    codes = {
        "cdi_mensal": 4391,
        "cdi_diario": 12
    }

    if not isinstance(indicators, list):
        indicators = [indicators]

    if any(ind not in codes for ind in indicators):
        print(f"Escolha um dos indicadores disponíveis: {', '.join(list(codes.keys()))}")

    if isinstance(start, str):
        start = datetime.strptime(start, "%Y-%m-%d")

    if isinstance(end, str):
        end = datetime.strptime(end, "%Y-%m-%d")

    try:

        if start + relativedelta(years=10) < end:

            meio = start + relativedelta(years=10)

            df_1 = sgs.get(
                {ind: codes[ind] for ind in indicators},
                start=start,
                end=meio
            )

            df_2 = sgs.get(
                {ind: codes[ind] for ind in indicators},
                start=meio,
                end=end
            )

            df = (
                pd.concat([df_1, df_2])
                .reset_index().drop_duplicates("Date")
            )

        else:

            df = sgs.get(
                {ind: codes[ind] for ind in indicators},
                start=start,
                end=end
            )

        return df
    
    except Exception as e:
        print(f"Erro na requisição ao SGS: {e}")