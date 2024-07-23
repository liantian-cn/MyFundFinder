#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

from lixinger import *


def main():
    fetch = query_json("cn/index", {})

    with open(DATA_DIR.joinpath("index.json"), 'w', encoding="utf-8") as f:
        json.dump(fetch, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
