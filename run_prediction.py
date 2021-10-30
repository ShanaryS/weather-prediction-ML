"""Use to run predictions with the models"""

from src.algorithms import main as main_


def main() -> None:
    """Main function"""

    main_(show=True, train=False, export=False)


if __name__ == '__main__':
    main()
