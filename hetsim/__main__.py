import sys

from .cli import main

if __name__ == "__main__":   # band-sweep worker processes import this module; they must not run the CLI
    sys.exit(main())
