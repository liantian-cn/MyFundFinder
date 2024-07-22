#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import pandas

from lixinger import *
import pandas as pd

logging.basicConfig(level=logging.INFO)


def calculate_percentile(df):
    df['pb_percentile'] = df['pb.mcw'].expanding().apply(lambda x: x.rank(method='min', pct=True).iloc[-1])
    df['pe_percentile'] = df['pe_ttm.mcw'].expanding().apply(lambda x: x.rank(method='min', pct=True).iloc[-1])
    df['dyr_percentile'] = df['dyr.mcw'].expanding().apply(lambda x: x.rank(method='min', pct=True).iloc[-1])
    df['ps_percentile'] = df['ps_ttm.mcw'].expanding().apply(lambda x: x.rank(method='min', pct=True).iloc[-1])
    # if 'fb' in df.columns:
    #     df['fb_percentile_1300'] = df['fb'].rolling(window=1300, min_periods=1).apply(lambda x: x.rank(method='min', pct=True).iloc[-1])
    # if 'sb' in df.columns:
    #     df['sb_percentile_1300'] = df['sb'].rolling(window=1300, min_periods=1).apply(lambda x: x.rank(method='min', pct=True).iloc[-1])
    # 1年约260个工作日
    # df['pb_percentile'] = df['pb'].rolling(window=2600, min_periods=1).apply(lambda x: x.rank(method='min', pct=True).iloc[-1])
    # df['pe_percentile'] = df['pe_ttm'].rolling(window=2600, min_periods=1).apply(lambda x: x.rank(method='min', pct=True).iloc[-1])
    # df['dry_percentile'] = df['dyr'].rolling(window=2600, min_periods=1).apply(lambda x: x.rank(method='min', pct=True).iloc[-1])
    # df['ps_percentile'] = df['ps_ttm'].rolling(window=2600, min_periods=1).apply(lambda x: x.rank(method='min', pct=True).iloc[-1])
    return df


def main():
    for file in AMENTAL_DIR.glob("*.json"):
        stock_code = file.stem
        with open(file, 'r', encoding="utf-8") as f:
            df = pandas.read_json(f, orient="records")

        df = calculate_percentile(df)

        with open(file, 'w', encoding="utf-8") as f:
            df.to_json(f, orient="records", indent=4)
        logging.info("Calculate percentile for %s", stock_code)


if __name__ == '__main__':
    main()
