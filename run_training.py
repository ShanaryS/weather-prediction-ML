"""Use to train the models"""

from algorithms import main as _main


def main() -> None:
    """Main function"""

    _main(show=True, train=True, export=True)


if __name__ == '__main__':
    main()
