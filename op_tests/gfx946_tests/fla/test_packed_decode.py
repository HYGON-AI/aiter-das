# SPDX-License-Identifier: MIT
"""Run with --suite full; see README.md for build, model and hardware setup.
Example: python test_packed_decode.py --expect-xcds 2 --json result.json
"""
from test_fla import main

if __name__ == "__main__":
    main("packed_decode")
