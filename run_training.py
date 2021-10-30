"""Use to train the models"""

from src.algorithms import main as main_


def main() -> None:
    """Main function"""

    main_(show=True, train=True, export=True)


if __name__ == '__main__':
    main()
