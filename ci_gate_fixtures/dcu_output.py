# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Intentional quality-gate fixtures. Do not merge or execute."""

import subprocess

value = 0


def wording_fixture():
    print("amd")
    print("dcu")
    print("amdsmi")
    print("abcdamd")
    print("xgmi")
    print("{}".format("amd"))


def ruff_fixture():
    print(value)
    value = 1


def semgrep_fixture():
    subprocess.run("printf gate-fixture", shell=True)
