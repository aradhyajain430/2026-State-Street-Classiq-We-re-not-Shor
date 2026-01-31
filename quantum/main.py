if __package__ is None:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from quantum.iqae_var import main


if __name__ == "__main__":
    main()
