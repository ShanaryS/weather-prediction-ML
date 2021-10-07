"""Use to run predictions with the models"""

from algorithms import main as _main


def main() -> None:
    """Main function"""

    _main(show=True, train=False, export=False)


if __name__ == '__main__':
    main()
