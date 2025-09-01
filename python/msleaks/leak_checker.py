# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

"""
在使用leak_checker离线泄漏分析前, 请确保文件中包含了泄漏分析用的MSTX打点信息,
打点需要至少有3处,用于确定想要检测泄漏的边界,
在这里需要先设置离线分析的CSV路径以及泄漏分析用的MSTX打点信息的标识字段。
"""

import csv
import sys
import os
import re
from collections import defaultdict

BYTE_TO_MB = 1024 * 1024


class LeakChecker:
    def __init__(self):
        self.device_events = defaultdict(list)  # 按设备ID分组存储事件
        self.error = None

    def _validate_path(self, csv_path):
        """验证路径安全性"""

        # 检查文件是否存在且可读
        if not os.path.isfile(csv_path):
            self.error = "Error: CSV File not found or not a regular file"
            return False
            
        if not os.access(csv_path, os.R_OK):
            self.error = "Error: Permission denied - CSV file is not readable"
            return False
            
        self._abs_path = csv_path  # 保存验证后的绝对路径
        return True

    def _read_file(self):
        """安全读取CSV文件"""
        try:
            with open(self._abs_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    device_id = row['Device Id']  
                    # 读取CSV文件按照设备号进行分组
                    self.device_events[device_id].append(row)
            return True
        except FileNotFoundError:
            self.error = "Error: File not found during access"
            return False
        except PermissionError:
            self.error = "Error: Permission denied during file access"
            return False
        except csv.Error as e:
            self.error = f"Error: CSV parsing error - {e}"
            return False
        except Exception as e:
            self.error = f"Error: Unexpected error - {e}"
            return False
    
    def _analyze_event(self, section1_start, section1_end, section2_end, allocations, event):
        # 开放态上只处理Hal的泄漏
        if event.get('Event Type') != 'HAL':
            return

        event_id = event.get('ID')
        # 超出第三个打点位置的直接跳过
        if int(event_id) > section2_end:
            return
        
        # 记录MALLOC事件
        if section1_start <= int(event_id) <= section1_end and event.get('Event') == 'MALLOC':
            alloc_addr = event.get('Ptr', '')
            match = re.search(r'size:(\d+)', event.get('Attr', ''))  # 匹配 "size:数字"
            alloc_size = int(match.group(1))/BYTE_TO_MB  # 提取数字并转为整数
            allocations[event_id] = {"addr": alloc_addr, "size": alloc_size}
            
        # 处理FREE事件
        if section1_start <= int(event_id) and event.get('Event') == 'FREE':
            free_addr = event.get('Ptr')
            # 不考虑double free 
            for index, info in list(allocations.items()):
                if free_addr == info['addr']:
                    del allocations[index]
    
    def _analyze_leaks(self, mstx_info, start_index):
        """
        对设备进行泄漏分析
        :param mstx_info: MSTX事件筛选条件,由用户指定
        :param start_index: 起始打点索引
        :return: 泄漏告警列表
        """
        leaks = []

        for device_id, events in self.device_events.items():
            
            if device_id == 'N/A':
                continue
            
            print(f"Starting to analyze data for device {device_id}...")
            # 按ID排序,需转为int
            events.sort(key=lambda x: int(x['ID']))
            
            # 查找指定内容的MSTX事件
            defined_mark_events = self._find_mstx_events(events, mstx_info)
            
            # 跟踪内存分配，allocation中维护申请未释放的内存
            allocations = {}

            if len(defined_mark_events) < 3:
                print(f"Device {device_id} has fewer than 3 data points. Please re-collect the data points.")
                continue
            else:
                # 获取三个打点位置的事件
                try:
                    point_a = defined_mark_events[start_index]
                    point_b = defined_mark_events[start_index+1]
                    point_c = defined_mark_events[start_index+2]
                except IndexError:
                    print(f"Start index {start_index} is out of bounds."
                    f" Device {device_id} has a total of {len(defined_mark_events)} MSTX events.")
                    continue
                
                # 获取三个打点ID
                section1_start = int(point_a['ID'])
                section1_end = int(point_b['ID'])
                section2_end = int(point_c['ID'])

                # 分析内存泄漏
                for event in events:
                    self._analyze_event(section1_start, section1_end, section2_end, allocations, event)

            if allocations:
                leaks.append(f"====== ERROR: Detected memory leaks on device {device_id} ======")
                for index, info in list(allocations.items()):
                    leaks.append(f"Direct Hal memory leak of {info['size']} Mb(s) at {hex(int(info['addr']))}"
                     f" in Index {index}.")
            else:
                leaks.append(f"No hal memory leaks detected on device {device_id}.")

        return leaks
    
    def _find_mstx_events(self, events, mstx_info):
        """查找符合条件的MSTX MARK事件"""
        defined_mark_events = []
        for event in events:
            if event.get('Event') == 'MSTX' and event.get('Event Type') == 'Mark' and event.get('Name') == mstx_info:
                defined_mark_events.append(event)

        return defined_mark_events
    
    def run(self, csv_path="", mstx_info="", start_index=0):
        """
        内存泄漏分析器对外接口
        
        参数:
            csv_path (str): CSV文件路径（相对路径）
            mstx_info (str): 内存接口信息配置
            start_index (int): 数据处理起始索引
        """
        if csv_path == '':
            print("ERROR: Please update the CSV file path to be read.")
            return

        if mstx_info == "":
            print("ERROR: Please update the MSTX information for the leak detection.")
            return
        
        # 安全校验和文件读取阶段
        if not self._validate_path(csv_path) or not self._read_file():
            if self.error:
                print("ERROR: Initialization failed:", self.error)
                return
        
        leaks = self._analyze_leaks(mstx_info, start_index)
        if leaks:
            print(f"\nLeak Analysis Results for the Custom Range:")
            for alert in leaks:
                print(f"{alert}")


_leakchecker = LeakChecker()