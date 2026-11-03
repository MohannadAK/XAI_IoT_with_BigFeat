"""Extract N-BaIoT's per-device RAR archives (p7zip on many systems lacks the RAR codec)."""
import glob
import os
import sys

import libarchive

root = sys.argv[1]
for rar in sorted(glob.glob(os.path.join(root, "**", "*.rar"), recursive=True)):
    dest = rar[:-4]
    os.makedirs(dest, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(dest)
    try:
        libarchive.extract_file(rar)
    finally:
        os.chdir(cwd)
    print("extracted", rar, flush=True)
    os.remove(rar)
