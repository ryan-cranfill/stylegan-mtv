import click

from src import add_src_to_sys_path
from src.explore import LatentSpaceExplorer

add_src_to_sys_path()


@click.group()
def cli():
    pass


@click.command()
@click.option('-p', '--prefix', default=None, help='Likes & dislikes prefix', type=str)
@click.option('-r', '--random_seed', default=False, help='Random seed', type=int)
@click.option('-m', '--model_name', default='wikiart-sg2', help='Stylegan model name', type=str)
@click.option('-s', '--save', is_flag=True, help='Save liked images')
def latent(prefix, random_seed, model_name, save):
    explorer = LatentSpaceExplorer(model_name, random_seed, save, prefix)
    explorer.explore()


cli.add_command(latent)


if __name__ == '__main__':
    cli()
