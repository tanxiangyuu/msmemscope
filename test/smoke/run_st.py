#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import os
import sys
import logging
from datetime import datetime

from src.utils.file_system import WorkingDir
from src.utils.result import Result
from src.utils.utils import ColorText
from src.utils.arg_parser import create_arg_parser

from src.test_suit.multirank_test import MultirankTestSuite
from src.test_suit.step_compare_test import StepCompareTestSuite
from src.test_suit.hijack_atb_test import HijackAtbTestSuite
from src.test_suit.mindspore_test import HijackMindsporeTestSuite
from src.test_suit.step_select_test import SelectOneStepTestSuite, SelectFirstStepTestSuite, \
    SelectLastStepTestSuite, SelectMultiStepTestSuite
from src.test_suit.watch_test import WatchATBHashTestSuite, WatchATBBinTestSuite, \
    WatchATENHashTestSuite, WatchATENBinTestSuite
from src.test_suit.llama2_7b_test import Llama2_7bTestSuite
from src.utils.env_checker import EnvChecker
from src.utils.symbol_checker import SymbolChecker


def report_summary(results: list[Result]) -> bool:
    suite_count = len(results)
    failed_count = len(list(filter(lambda r: not r.success, results)))
    success = failed_count == 0

    print(f"{ColorText.border} reduce result from {suite_count} test suite(s)")
    if success:
        print(f"{ColorText.run_ok} all test suites pass")
    else:
        print(f"{ColorText.run_failed} {failed_count} suite(s) failed")
    print(f"{ColorText.border} reduce result from {suite_count} test suite(s)")
    return success


def chmod_recursive(path, permission):
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), permission)
        for f in files:
            os.chmod(os.path.join(root, f), permission)


def read_config(file_path):
    config = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # 跳过注释行和空行
                if not line or line.startswith('#'):
                    continue
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"config.txt not found!")
        return None
    return config


def run_tests(working_dir: str, params) -> bool:
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)
    
    # config = read_config('./config.txt')
    
    chmod_recursive("msmemscope", 0o777)
    multirank_cmd_command = ["../../msmemscope/output/bin/msmemscope", "bash", "../../testfile/scripts/test_multirank_cmd.sh",
        "--log-level=info", "--call-stack=c,python", "--level=0,1"]
    multirank_api_command = ["bash", "../../testfile/scripts/test_multirank_api.sh"]
    # llama2_7b_cmd = ["../../msmemscope/output/bin/msmemscope", "bash", config.get('llama2_7b_sh_path'), "--log-level=info"]
    base_file_path = "../../testfile/csvfile/memscope_dump_20250428113810.csv"
    compare_file_path = "../../testfile/csvfile/memscope_dump_20250428113811.csv"
    step_compare_cmd = ["../../msmemscope/output/bin/msmemscope", 
        f"--input={base_file_path},{compare_file_path}", "--compare", "--log-level=info", "--level=1"]
    hijack_mindspore_cmd_command = ["../../msmemscope/output/bin/msmemscope", "bash", "../../testfile/scripts/test_mindspore_cmd.sh",
        "--log-level=info", "--level=0,1"]
    hijack_mindspore_api_command = ["bash", "../../testfile/scripts/test_mindspore_api.sh"]
    hijack_atb_cmd = ["../../msmemscope/output/bin/msmemscope", "bash", "../../testfile/scripts/test_hijack_atb.sh",
        "--log-level=info", "--level=0,1", "--events=launch,access"]
    select_one_step_cmd = ["../../msmemscope/output/bin/msmemscope", "python", "../../testfile/scripts/select_steps/select_one_steps.py"]
    select_first_step_cmd = ["../../msmemscope/output/bin/msmemscope", "python", "../../testfile/scripts/select_steps/select_one_steps.py",
        "--steps=1"]
    select_last_step_cmd = ["../../msmemscope/output/bin/msmemscope", "python", "../../testfile/scripts/select_steps/select_one_steps.py",
        "--steps=5"]
    select_multi_step_cmd = ["../../msmemscope/output/bin/msmemscope", "python", "../../testfile/scripts/select_steps/select_one_steps.py",
        "--steps=1,2,5"]
    watch_atb_hash_cmd = ["../../msmemscope/output/bin/msmemscope", "../../testfile/scripts/watch/op_atb_0", "--level=0,1",
        "--watch=0/0_ElewiseOperation/0_AddI32Kernel,0/0_ElewiseOperation"]
    watch_atb_bin_cmd = ["../../msmemscope/output/bin/msmemscope", "../../testfile/scripts/watch/op_atb_0", "--level=0,1",
        "--watch=0/0_ElewiseOperation/0_AddI32Kernel,0/0_ElewiseOperation,full-content"]
    watch_aten_hash_cmd = ["../../msmemscope/output/bin/msmemscope", "bash", "../../testfile/scripts/watch/watch_aten_hash_cmd.sh",
        "--watch=torch._ops.aten.mse_loss.default,torch._ops.aten.mse_loss_backward.default"]
    watch_aten_bin_cmd = ["../../msmemscope/output/bin/msmemscope", "bash", "../../testfile/scripts/watch/watch_aten_bin_cmd.sh",
        "--watch=torch._ops.aten.mse_loss.default,torch._ops.aten.mse_loss_backward.default,full-content"]
    test_suites = [
        MultirankTestSuite("msmemscope_multirank_cmd_test", params, "check_multirank_cmd", multirank_cmd_command, 100),
        MultirankTestSuite("msmemscope_multirank_api_test", params, "check_multirank_api", multirank_api_command, 100),
        StepCompareTestSuite("step_compare_test", params, "check_step_compare", step_compare_cmd, 100),
        HijackAtbTestSuite("hijack_atb_test", params, "check_hijack_atb", hijack_atb_cmd, 100),
        HijackMindsporeTestSuite("hijack_mindspore_cmd_test", params, "check_hijack_mindspore_cmd", hijack_mindspore_cmd_command, 1000),
        HijackMindsporeTestSuite("hijack_mindspore_api_test", params, "check_hijack_mindspore_api", hijack_mindspore_api_command, 1000),
        WatchATBHashTestSuite("watch_atb_test", params, "check_watch_atb_hash", watch_atb_hash_cmd, 100),
        WatchATBBinTestSuite("watch_atb_test", params, "check_watch_atb_bin", watch_atb_bin_cmd, 100),
        WatchATENHashTestSuite("watch_aten_test", params, "check_watch_aten_hash", watch_aten_hash_cmd, 100),
        WatchATENBinTestSuite("watch_aten_test", params, "check_watch_aten_bin", watch_aten_bin_cmd, 100),
    ]
    
    if params.llama2_7b:
        test_suites.append(Llama2_7bTestSuite("llama2_7b_test", params, "check_llama2_7b", llama2_7b_cmd, 1000))

    if params.select_steps_case:
        test_suites.append(SelectOneStepTestSuite("select_one_step_test", params, "check_select_one_step", select_one_step_cmd, 100))
        test_suites.append(SelectFirstStepTestSuite("select_first_step_test", params, "check_select_first_step", select_first_step_cmd, 100))
        test_suites.append(SelectLastStepTestSuite("select_last_step_test", params, "select_last_step_cmd", select_last_step_cmd, 100))
        test_suites.append(SelectMultiStepTestSuite("select_multi_step_test", params, "select_multi_step_cmd", select_multi_step_cmd, 100))
    
    results = []
    with WorkingDir(working_dir):
        for test_suite in test_suites:
            with test_suite:
                results.append(test_suite.run())

    return report_summary(results)


def log_name():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"./st_log/msmemscope_run_st_{timestamp}.log"


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args(sys.argv[1:])
    if not os.path.exists('st_log'):
        os.mkdir('st_log')
    logging.basicConfig(filename=log_name(),
                        level=logging.DEBUG,
                        format="<%(asctime)s> [%(levelname)s] %(message)s")

    env_path = "/usr/local/Ascend"
    checker = EnvChecker(env_path)
    logging.debug(f"environment {checker.env}")
    if not checker.env:
        sys.exit(os.EX_SOFTWARE)

    if run_tests("workbench", args):
        sys.exit(os.EX_OK)
    else:
        sys.exit(os.EX_SOFTWARE)


if __name__ == "__main__":
    main()
