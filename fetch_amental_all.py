#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
from datetime import datetime, timedelta

import pandas as pd

from lixinger import *

logging.basicConfig(level=logging.INFO)


def get_dates_ranges(launch_datetime, end_datetime):
    """
    根据开始日期和发射日期计算日期范围列表。

    参数:
    start_datetime: datetime对象，表示开始日期。
    launch_datetime: datetime对象，表示发射日期。

    返回:
    一个包含日期范围元组的列表，每个元组包含两个格式为"%Y-%m-%d"的字符串，分别表示范围的开始和结束日期。

    抛出:
    ValueError: 如果发射日期晚于开始日期，则抛出此异常。
    """
    # 检查发射日期是否早于开始日期，如果不是，则抛出ValueError
    if launch_datetime > end_datetime:
        raise ValueError("Launch date must be before start date")

    date_ranges = []
    # 当发射日期小于开始日期时，循环继续
    while launch_datetime < end_datetime:
        # 计算结束日期，确保不超过开始日期，且最多增加10年
        _end_datetime = min(launch_datetime + timedelta(days=365 * 9), end_datetime)
        # 将开始日期和结束日期格式化为字符串，并添加到日期范围列表中
        date_ranges.append((launch_datetime.strftime("%Y-%m-%d"), _end_datetime.strftime("%Y-%m-%d")))
        # 将发射日期更新为结束日期的下一天，为下一次循环做准备
        launch_datetime = _end_datetime + timedelta(days=1)

    return date_ranges


def fetch_data(start_date, end_date, stock_code):
    fetch = query_json(url_suffix="cn/index/fundamental",
                       query_params={
                           "startDate": start_date,
                           "endDate": end_date,
                           "stockCodes": [stock_code, ],
                           "metricsList": [
                               "pe_ttm.mcw",
                               "ps_ttm.mcw",
                               "pb.mcw",
                               "dyr.mcw",
                               "mc",
                               "cp",
                               # "cpc",
                               "to_r",
                               # "sb",
                               # "fb"
                           ]
                       }
                       )
    if fetch['message'] != "success":
        raise Exception
    return fetch["data"]


def fetch_all_data(launch_datetime, end_datetime, stock_code):
    date_ranges = get_dates_ranges(launch_datetime, end_datetime)
    all_data = []

    for start, end in date_ranges:
        data = fetch_data(start, end, stock_code)
        all_data.extend(data)
    # print(all_data)
    all_data.sort(key=lambda x: x['date'])
    return all_data


def main():
    with open(DATA_DIR.joinpath("index.json"), 'r', encoding="utf-8") as f:
        _ = json.load(f)
        indexes = _["data"]

    for index in indexes:
        launch_datetime = datetime.fromisoformat(index["launchDate"])
        stock_code = index["stockCode"]

        fetch = fetch_all_data(
            launch_datetime=launch_datetime,
            end_datetime=datetime.now(launch_datetime.tzinfo),
            stock_code=stock_code)

        df = pd.json_normalize(fetch)
        df['date'] = pd.to_datetime(df['date']).dt.tz_convert(None)
        with open(AMENTAL_DIR.joinpath(f"{stock_code}.json"), 'w', encoding="utf-8") as f:
            df.to_json(f, orient="records", indent=4)
        logging.info(f"fetch {stock_code} data success")
    logging.info("fetch all data success")


if __name__ == '__main__':
    main()
