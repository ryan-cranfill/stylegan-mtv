from src import add_src_to_sys_path
from src.explore import LatentSpaceExplorer

add_src_to_sys_path()


def main():
    explorer = LatentSpaceExplorer()
    explorer.explore()


if __name__ == '__main__':
    main()
