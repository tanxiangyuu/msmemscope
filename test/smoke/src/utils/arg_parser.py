#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from argparse import ArgumentParser


def create_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="msmemscope test framework",
                            description="test framework for msmemscope test cases")
    parser.add_argument("-f", "--filter",
                        help="filter test cases to run. regex is supported")
    parser.add_argument("-l", "--llama2_7b", action='store_true', help="enable modelLink test case")
    parser.add_argument("-s", "--select_steps_case", action='store_true', help="enable select steps test cases")
    return parser