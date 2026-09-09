"""CLI launcher for SGLang server with native EXL3 quantization support."""

import sys
from sglang_exl3 import register


def main():
    register()
    from sglang.launch_server import main as sglang_main

    sglang_main()


if __name__ == "__main__":
    main()
