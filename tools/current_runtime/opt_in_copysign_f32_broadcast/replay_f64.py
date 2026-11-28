#!/usr/bin/env python3
"""Replay the frozen F64 broadcast differential on the combined image."""
from hashlib import sha256
from pathlib import Path
import runpy

FILE = Path(__file__).resolve().parent.parent/"opt_in_copysign_f64_broadcast/qualify.py"
SOURCE = "83687ed8713fd9d07353eea5dfc065fabc2266c04c91d36fd8748990fc25ff10"
OLD = "5537162e8cfe8edd6e2a30474e7d3c001a76c5731c5ff94c1930bb27adb544a5"
NEW = "7fb4e5a09887dee8d9ec6b60c0f19b7c9ea86db64e9f423952305274d818b99a"


def main():
    if sha256(FILE.read_bytes()).hexdigest() != SOURCE:
        raise ValueError("expected exact #96 F64 qualification")
    referee = runpy.run_path(str(FILE))["main"]
    env = referee.__globals__
    if env["CANDIDATE"] != OLD:
        raise ValueError("expected predecessor's candidate image pin")
    env["CANDIDATE"] = NEW
    referee()


if __name__ == "__main__":
    main()
