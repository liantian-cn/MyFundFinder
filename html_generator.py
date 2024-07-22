#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from lixinger import *
from jinja2 import Environment, FileSystemLoader
import pandas


def float_to_percentile(value):
    if value is None:
        return "--"
    return f"{value*100:.2f}%"

def float_to_2f(value):
    if value is None:
        return "--"
    return f"{value:.2f}"

class Index():

    def __init__(self, attributes):
        if not isinstance(attributes, dict):
            raise TypeError("attributes must be a dict")
        if not attributes.get("stockCode"):
            raise ValueError("stockCode must be set")
        for key, value in attributes.items():
            setattr(self, key, value)
        self.stockCode = attributes["stockCode"]
        with open(AMENTAL_DIR.joinpath(f"{self.stockCode}.json"), 'r', encoding="utf-8") as f:
            df = pandas.read_json(f, orient="records")
            self.update = df['date'].iloc[-1]

            self.pe = df['pe_ttm.mcw'].iloc[-1]
            self.pe_percentile = df['pe_percentile'].iloc[-1]

            self.pe_display = float_to_2f(self.pe)
            self.pe_percentile_display = float_to_percentile(self.pe_percentile)

            self.pb = df['pb.mcw'].iloc[-1]
            self.pb_percentile = df['pb_percentile'].iloc[-1]

            self.pb_display = float_to_2f(self.pb)
            self.pb_percentile_display = float_to_percentile(self.pb_percentile)

            self.dyr = df['dyr.mcw'].iloc[-1]
            self.dyr_percentile = df['dyr_percentile'].iloc[-1]

            self.dyr_display = float_to_percentile(self.dyr)
            self.dyr_percentile_display = float_to_percentile(self.dyr_percentile)

            self.ps = df['ps_ttm.mcw'].iloc[-1]
            self.ps_percentile = df['ps_percentile'].iloc[-1]

            self.ps_display = float_to_2f(self.ps)
            self.ps_percentile_display = float_to_percentile(self.ps_percentile)

            if 'to_r' in df.columns:
                self.to_r = df['to_r'].iloc[-1]
            else:
                self.to_r = None

            self.to_r_display = float_to_percentile(self.to_r)

            self.mc = df['mc'].iloc[-1]
            self.mc_display = f"{self.mc/100000000:.2f}"

            if 'fb' in df.columns:
                self.fb_percentile = df['fb_percentile_1300'].iloc[-2]
            else:
                self.fb_percentile = None

            self.fb_display = float_to_percentile(self.fb_percentile)

            if 'sb' in df.columns:
                self.sb_percentile = df['sb_percentile_1300'].iloc[-2]
            else:
                self.sb_percentile = None

            self.sb_display = float_to_percentile(self.sb_percentile)

            self.total_days = df.shape[0]

        self.etf = []
        with open(FUND_DIR.joinpath(f"{self.stockCode}.json"), 'r', encoding="utf-8") as f:
            funds = json.load(f)
            funds = funds["data"]
            for fund in funds:
                if fund["exchange"] in ["sh", "sz"]:
                    self.etf.append(fund["stockCode"])


def main():
    indexes = []
    with open(DATA_DIR.joinpath("index.json"), 'r', encoding="utf-8") as f:
        _ = json.load(f)
        for i in _["data"]:
            index = Index(i)
            indexes.append(index)

    # 创建一个Jinja2环境
    env = Environment(loader=FileSystemLoader('.'))
    # 加载模板
    template = env.get_template('template.html')

    # 排序

    sorted_list = sorted(indexes, key=lambda x: x.pb_percentile * x.pe_percentile)

    # 渲染模板
    output = template.render(indexes=sorted_list)

    # 将渲染后的结果写入输出文件
    with open('output/index.html', 'w', encoding="utf-8") as file:
        file.write(output)


if __name__ == '__main__':
    main()
