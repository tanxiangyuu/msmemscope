#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.


from termcolor import colored


class ColorText:
    border = colored("[----------]", "green")
    run_test = colored("[ RUN      ]", "green")
    run_ok = colored("[       OK ]", "green")
    run_failed = colored("[  FAILED  ]", "red")
    run_warn = colored("[   WARN   ]", "yellow")
    run_list = colored("[   LIST   ]", "green")


class PrintBorder:
    def __init__(self, enter_text, exit_text=None):
        self._enter_text = enter_text
        self._exit_text = exit_text if exit_text is not None else enter_text

    def __enter__(self):
        print(f"{ColorText.border} {self._enter_text}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"{ColorText.border} {self._exit_text}\n")
