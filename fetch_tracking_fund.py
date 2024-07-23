#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging

from lixinger import *

logging.basicConfig(level=logging.INFO)


def main():
    with open(DATA_DIR.joinpath("index.json"), 'r', encoding="utf-8") as f:
        _ = json.load(f)
        indexes = _["data"]

    for index in indexes:
        stock_code = index["stockCode"]
        fetch = query_json(
            url_suffix="cn/index/tracking-fund",
            query_params={"stockCode": stock_code}
        )

        with open(FUND_DIR.joinpath(f"{stock_code}.json"), 'w', encoding="utf-8") as f:
            json.dump(fetch, f, ensure_ascii=False, indent=4)
        logging.info(f"{stock_code} Done")
    logging.info("Done")


if __name__ == '__main__':
    main()
