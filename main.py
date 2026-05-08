import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from nte.cli import main, set_seed


if __name__ == "__main__":
    set_seed()
    main()
