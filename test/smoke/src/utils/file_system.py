#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import os


class WorkingDir:
    def __init__(self, path):
        self._path = path

    def __enter__(self):
        self.current = os.getcwd()
        os.chdir(self._path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.current)

