# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import subprocess

if __name__ == "__main__":
    cmakeCmd = ["cmake", ".."]
    ret = subprocess.run(cmakeCmd, capture_output=False)
    if ret.returncode != 0:
        exit(ret.returncode)

    ret = subprocess.run(["make"], capture_output=False)
    exit(ret.returncode)
