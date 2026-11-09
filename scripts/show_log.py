import re
import sys
from pathlib import Path

log = Path(sys.argv[1])
pat = sys.argv[2] if len(sys.argv) > 2 else "loss"
for line in log.read_text(errors="replace").splitlines():
    if re.search(pat, line):
        print(line[:400])
