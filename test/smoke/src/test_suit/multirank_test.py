#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import os
import shutil
import logging
import re
import pandas as pd
from packaging import version
import torch

from .base_test import BaseTest, TestSuite
from ..utils.result import Result
from ..utils.file_system import WorkingDir
from ..utils.utils import ColorText


class MultirankTestSuite(TestSuite):
    def __init__(self, name: str, config, work_path: str, cmd: str, max_time):
        super().__init__(name, config, work_path, cmd, max_time)

        # 传入name区分命令行和api
        test_cases = [
            MultirankTestCase("check_dump", name, work_path, ""),
            #MultirankTestCase("check_log", work_path, ""),
            #MultirankAnaLyzerModuleTestCase("check_npu_leaks", work_path, ""),
            #MultirankAnaLyzerModuleTestCase("check_leaks_warning", work_path, ""),
            #MultirankAnaLyzerModuleTestCase("check_gap_analysis", work_path, ""),
        ]
        _ = list(map(self.register, test_cases))

    def __str__(self):
        return f"msmemscope test suite. suite name: {self.name}, " \
               f"suite work path: {self._work_path}"

    def set_up(self):
        super().set_up()

        if not os.path.exists(self._work_path):
            os.mkdir(self._work_path)

        with WorkingDir(self._work_path):
            if os.path.exists('memscopeDumpResults'):
                shutil.rmtree('memscopeDumpResults')
            log_files = [name for name in os.listdir(".") if name.endswith('.log')]
            for file in log_files:
                os.remove(file)
            json_files = [name for name in os.listdir(".") if name.endswith('.json')]
            for file in json_files:
                os.remove(file)
    def tear_down(self):
        super().tear_down()


class MultirankTestCase(BaseTest):
    def __init__(self, name: str, case_name: str, real_path: str, golden_path: str):
        super().__init__(name)
        self.case_name = case_name
        self._golden_path = golden_path
        self._real_path = real_path

    def __str__(self):
        return f"case name: {self.name}, " \
               f"case path: {self._real_path}"

    def comp_memscope_csv_contents(self, file_paths, column):

        # 由于多卡冒烟涉及多个文件，必须合并后统计
        dfs = []
        for file in file_paths:
            try:
                # 读取所有CSV文件,进行合并
                df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                logging.error(f"Error reading {file}: {str(e)}")
        
        # 合并所有 DataFrame（按行叠加）
        if dfs:
            data = pd.concat(dfs, ignore_index=True)
        
        Event_counts = data['Event'].value_counts()
        Event_type_counts = data['Event Type'].value_counts()

        if list(data.columns) != column.split(','):
            logging.error("the sum data column of %s is error, please check your dump file")
            return Result(False, [column], [data.columns])

        if version.parse(torch.__version__) < version.parse("2.3.0"):
            min_threshold = 15000
            max_threshold = 17000

            if len(data) < min_threshold or len(data) > max_threshold:
                logging.error("the sum data length of %s is error")
                return Result(False, ["min: ", min_threshold, "max: ", max_threshold], [len(data)])

        # 命令行和API校验条件有些不一致
        if self.case_name == "msmemscope_multirank_cmd_test":
            system_num = 6
            mstx_num = 202
            op_threshold = {"min":16000, "max":16500}
            kernel_threshold = {"min":16000, "max":16500}
            hal_threshold = {"min":250, "max":350}
            pta_threshold = {"min":9200, "max":9300}
        else:
            system_num = 9                                  # API不一致
            mstx_num = 202
            op_threshold = {"min":16000, "max":16500}
            kernel_threshold = {"min":16000, "max":16500}
            hal_threshold = {"min":200, "max":350}          # API不一致
            pta_threshold = {"min":8700, "max":8800}        # API不一致

            if Event_counts.get('SYSTEM', None) is None:
                logging.error("SYSTEM key not found in Event_counts")
                return Result(False, ["SYSTEM key not found", -1], [-1])
            if Event_counts['SYSTEM'] != system_num:
                logging.error("the length of %s is error", file)
                return Result(False, ["SYSTEM: ", system_num], [Event_counts['SYSTEM']])
            
            if Event_counts.get('MSTX', None) is None:
                logging.error("MSTX key not found in Event_counts")
                return Result(False, ["MSTX key not found", -1], [-1])
            if Event_counts['MSTX'] != mstx_num:
                logging.error("the length of %s is error", file)
                return Result(False, ["MSTX: ", mstx_num], [Event_counts['MSTX']])

            if Event_counts.get('KERNEL_LAUNCH', None) is None:
                logging.error("KERNEL_LAUNCH key not found in Event_counts")
                return Result(False, ["KERNEL_LAUNCH key not found", -1], [-1])
            if Event_counts['KERNEL_LAUNCH'] < kernel_threshold['min'] or \
                Event_counts['KERNEL_LAUNCH'] > kernel_threshold['max']:
                logging.error("the length of %s is error", file)
                return Result(False, ["KERNEL_LAUNCH min: ", kernel_threshold['min'],
                    "KERNEL_LAUNCH max: ", kernel_threshold['max']], [Event_counts['KERNEL_LAUNCH']])

            if Event_type_counts.get('HAL', None) is None:
                logging.error("HAL key not found in Event_type_counts")
                return Result(False, ["HAL key not found", -1], [-1])
            if Event_type_counts['HAL'] < hal_threshold['min'] or Event_type_counts['HAL'] > hal_threshold['max']:
                logging.error("the length of %s is error", file)
                return Result(False, ["HAL min: ", hal_threshold['min'], "HAL max: ", hal_threshold['max']],
                    [Event_type_counts['HAL']])

            if Event_type_counts.get('PTA', None) is None:
                logging.error("PTA key not found in Event_type_counts")
                return Result(False, ["PTA key not found", -1], [-1])
            if Event_type_counts['PTA'] < pta_threshold['min'] or Event_type_counts['PTA'] > pta_threshold['max']:
                logging.error("the length of %s is error", file)
                return Result(False, ["PTA min: ", pta_threshold['min'], "PTA max: ", pta_threshold['max']],
                    [Event_type_counts['PTA']])

            if Event_counts.get('OP_LAUNCH', None) is None:
                logging.error("OP_LAUNCH key not found in Event_counts")
                return Result(False, ["OP_LAUNCH key not found", -1], [-1])
            if Event_counts['OP_LAUNCH'] < op_threshold['min'] or Event_counts['OP_LAUNCH'] > op_threshold['max']:
                logging.error("the length of %s is error", file)
                return Result(False, ["OP_LAUNCH min: ", op_threshold['min'], "OP_LAUNCH max: ", op_threshold['max']],
                    [Event_counts['OP_LAUNCH']])

        return Result(True, [], [])
    
    @staticmethod
    def count_substring(data, phase, name) -> int:
        count = 0
        for entry in data:
            if entry.get("ph") == phase and name in entry.get("name"):
                count += 1
        return count

    def compare_log(self):
        logging.info("checking log...")
        FILE_GEN_COUNT = 1
        FILE_GEN_DIR = self._real_path
        LEAK_COUNT, MSTX_START_COUNT, STEP_INNER_COUNT = 20, 4, 100
        if not os.path.exists(FILE_GEN_DIR):
            logging.error("directory %s not exist", FILE_GEN_DIR)
            return Result(False, [], [])
        new_log_files = [name for name in os.listdir(FILE_GEN_DIR) if name.endswith('.log') and name[0] == 'm']
    
        if len(new_log_files) != FILE_GEN_COUNT:
            logging.error("Failed to generate %d log files", FILE_GEN_COUNT)
            return Result(False, [FILE_GEN_COUNT], [len(new_log_files)])

        real_log_file = os.path.join(FILE_GEN_DIR, new_log_files[0])
        with open(real_log_file, 'r') as f:

            file_text = f.read()
            leak_count = file_text.count("Leak memory in Malloc operator")

            if leak_count < LEAK_COUNT:
                logging.error("msmemscope detect failed")
                return Result(False, [LEAK_COUNT], [leak_count])
            mstx_start_count = file_text.count("mstxMarkA")
            step_inner_count = file_text.count("step start") 
            if mstx_start_count != MSTX_START_COUNT or step_inner_count != STEP_INNER_COUNT:
                logging.error("mstx detect failed")
                return Result(False, ["mstx_start_count", MSTX_START_COUNT, "step_inner_count", STEP_INNER_COUNT], [mstx_start_count, step_inner_count])

        logging.info("check finish")
        return Result(True, [], [])

    def compare_memscope_csv(self):

        FILE_GEN_COUNT = 3
        FILE_GEN_DIR = os.path.join(self._real_path, 'memscopeDumpResults')
        logging.info("checking csv...")
        if not os.path.exists(FILE_GEN_DIR):
            logging.error("directory %s not exist", FILE_GEN_DIR)
            return Result(False, [], [])

        # 递归查找所有子目录中的 CSV 文件,保存名称和位置
        new_csv_files_names = []
        new_csv_files_paths = []
        for root, dirs, files in os.walk(FILE_GEN_DIR):
            for file in files:
                if file.endswith('.csv') and file.startswith('memscope_dump'):
                    # 构建完整文件路径并添加到列表
                    full_path = os.path.join(root, file)
                    new_csv_files_names.append(file)
                    new_csv_files_paths.append(full_path)

        # 多卡冒烟当前落盘3个csv文件
        if len(new_csv_files_names) != FILE_GEN_COUNT:
            logging.error("Failed to generate %d CSV files", FILE_GEN_COUNT)
            return Result(False, [FILE_GEN_COUNT], [len(new_csv_files_names)])
        
        for i in range(FILE_GEN_COUNT):
            if not re.match('memscope_dump_\d{1,20}\.csv', new_csv_files_names[i]):
                logging.error("A CSV file matching naming convention memscope_dump could not be found")
                return Result(False, [], [])

        column = ("ID,Event,Event Type,Name,Timestamp(ns),Process Id,Thread Id,Device Id,"
                "Ptr,Attr,Call Stack(Python),Call Stack(C)")
        
        result = self.comp_memscope_csv_contents(new_csv_files_paths, column)
        if not result.success:
            return result
        logging.info("check finish")
        return Result(True, [], [])

    def run(self) -> Result:
        super().run()
        logging.debug(f"run {self}")
        print(f"{ColorText.run_test} {self}")

        result = Result(False, [], [])
        if self._name == "check_log":
            result = self.compare_log()
        elif self._name == "check_dump":
            result = self.compare_memscope_csv()
        self.report(result)
        return result

    def set_up(self):
        super().set_up()

    def tear_down(self):
        super().tear_down()

class MultirankAnaLyzerModuleTestCase(BaseTest):
    def __init__(self, name: str, real_path: str, golden_path: str):
        super().__init__(name)
        self._golden_path = golden_path
        self._real_path = real_path

    def __str__(self):
        return f"case name: {self.name}, " \
               f"case path: {self._real_path}"

    def check_npu_leaks(self):
        logging.info("checking npu leaks...")
        FILE_GEN_COUNT = 1
        FILE_GEN_DIR = self._real_path
        NPU_LEAK_COUNT = 48
        if not os.path.exists(FILE_GEN_DIR):
            logging.error("directory %s not exist", FILE_GEN_DIR)
            return Result(False, [], [])
        new_log_files = [name for name in os.listdir(FILE_GEN_DIR) if name.endswith('.log') and name[0] == 'm']
    
        if len(new_log_files) != FILE_GEN_COUNT:
            logging.error("Failed to generate %d log files", FILE_GEN_COUNT)
            return Result(False, [FILE_GEN_COUNT], [len(new_log_files)])

        real_log_file = os.path.join(FILE_GEN_DIR, new_log_files[0])
        with open(real_log_file, 'r') as f:

            file_text = f.read()
            npu_leak_count = file_text.count("------leaks")

            if npu_leak_count != NPU_LEAK_COUNT:
                logging.error("npu leaks detect failed")
                return Result(False, [NPU_LEAK_COUNT], [npu_leak_count])

        logging.info("check finish")
        return Result(True, [], [])

    def check_leaks_warning(self):
        logging.info("checking leaks warning...")
        FILE_GEN_COUNT = 1
        FILE_GEN_DIR = self._real_path
        NPU_LEAK_WARNING_COUNT = 2300
        if not os.path.exists(FILE_GEN_DIR):
            logging.error("directory %s not exist", FILE_GEN_DIR)
            return Result(False, [], [])
        new_log_files = [name for name in os.listdir(FILE_GEN_DIR) if name.endswith('.log') and name[0] == 'm']
    
        if len(new_log_files) != FILE_GEN_COUNT:
            logging.error("Failed to generate %d log files", FILE_GEN_COUNT)
            return Result(False, [FILE_GEN_COUNT], [len(new_log_files)])
        real_log_file = os.path.join(FILE_GEN_DIR, new_log_files[0])
        with open(real_log_file, 'r') as f:

            file_text = f.read()
            npu_leak_count = file_text.count("Please check if there is leaks in Pytorch Caching memory pool.")

            if npu_leak_count <= NPU_LEAK_WARNING_COUNT:
                logging.error("leaks warning detect failed")
                return Result(False, [NPU_LEAK_WARNING_COUNT], [npu_leak_count])

        logging.info("check finish")
        return Result(True, [], [])
    
    def check_gap_analysis(self):
        logging.info("checking gap analysis...")
        FILE_GEN_DIR = self._real_path
        DEVICE_COUNT = 2
        MIN_GAP_STEP = 2
        MAX_GAP_STEP_DEVICE_0 = 3
        MAX_GAP_STEP_DEVICE_1 = 49
        if not os.path.exists(FILE_GEN_DIR):
            logging.error("directory %s not exist", FILE_GEN_DIR)
            return Result(False, [], [])
        new_output_files = [name for name in os.listdir(FILE_GEN_DIR) if name.endswith('.txt') and name[0] == 'o']

        real_output_file = os.path.join(FILE_GEN_DIR, new_output_files[0])
        device_count = 0
        with open(real_output_file, 'r') as f:
            for line in f:
                if line.startswith("MinGap"):
                    device_count += 1
                    parts = line.strip().split()
                    if parts:
                        min_gap_step = int(parts[-1])
                        
                        if min_gap_step != MIN_GAP_STEP:
                            logging.error("MinGap step Error.")
                            return Result(False, [MIN_GAP_STEP], [min_gap_step])
                    
                    else:
                        logging.error("GapAnalysis error .")
                        return Result(False, [], [])
                
                if line.startswith("MaxGap"):
                    parts = line.strip().split()
                    if parts:
                        max_gap_step = int(parts[-1])
                        alloc_percent = float(parts[1])

                        if alloc_percent > 50.0:
                            if max_gap_step != MAX_GAP_STEP_DEVICE_1:
                                logging.error("Device1 MaxGap step Error.")
                                return Result(False, [MAX_GAP_STEP_DEVICE_1], [max_gap_step])
                        else:
                            if max_gap_step != MAX_GAP_STEP_DEVICE_0:
                                logging.error("Device0 MaxGap step Error.")
                                return Result(False, [MAX_GAP_STEP_DEVICE_0], [max_gap_step])                            
                    
                    else:
                        logging.error("GapAnalysis error .")
                        return Result(False, [], [])
            
            if device_count != DEVICE_COUNT:
                logging.error("Device count error.")
                return Result(False, [DEVICE_COUNT], [device_count])
        
        return Result(True, [], [])

    def run(self) -> Result:
        super().run()
        logging.debug(f"run {self}")
        print(f"{ColorText.run_test} {self}")
        
        #具体测试逻辑
        result = Result(False, [], [])
        if self._name == "check_npu_leaks":
            result = self.check_npu_leaks()
        if self._name == "check_leaks_warning":
            result = self.check_leaks_warning()
        if self._name == "check_gap_analysis":
            result = self.check_gap_analysis()
        self.report(result)
        return result

    def set_up(self):
        super().set_up()

    def tear_down(self):
        super().tear_down()