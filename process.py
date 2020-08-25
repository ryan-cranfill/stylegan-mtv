import click

from utils import make_output_path
from src import add_src_to_sys_path, SpectrogramOfflineProcessor, InterpolationOfflineProcessor,\
    SpectrogramInterpolationOfflineProcessor

add_src_to_sys_path()

@click.group()
def cli():
    pass


def common_options(function):
    function = click.option('--model_name', default='wikiart', help='Model name (without .pkl extension)', type=str)(function)
    function = click.option('--fps', default=24, help='frames per second', type=int)(function)
    function = click.option('--random_seed', default=False, help='random seed', type=int)(function)
    function = click.option('--start', default=0, help='Start time', type=int)(function)
    function = click.option('--duration', default=None, help='Duration of video to make', type=int)(function)
    function = click.option('--sr', default=None, help='sample rate', type=int)(function)
    function = click.option('--frame_chunk_size', default=500, help='Number of frames to batch before writing to disk', type=int)(function)
    function = click.option('--no_write', is_flag=True, help='Do not write out video.')(function)
    function = click.option('--output_path', default='', help='Output path', type=str)(function)
    function = click.argument('input_path')(function)
    return function


@click.command()
@common_options
@click.option('--window_size', default=5, help='Window size', type=int)
@click.option('--displacement_factor', default=0.1, help='Displacement factor', type=float)
def spectro(model_name, fps, random_seed, start, duration, sr, window_size, displacement_factor,
            frame_chunk_size, no_write, input_path, output_path):
    if not output_path:
        output_path = make_output_path(input_path)
    print('================ PARAMETERS')
    print(model_name, fps, random_seed, input_path, output_path, duration, )

    # todo: add auto output file name here

    processor = SpectrogramOfflineProcessor(model_name, fps, random_seed, frame_chunk_size)
    processor.process_file(input_path, output_path, start, duration, sr, not no_write, window_size, displacement_factor)


@click.command()
@common_options
@click.option('--n_points', default=3, help='Number of points to interpolate between', type=int)
def interp(model_name, n_points, fps, random_seed, start, duration, sr, frame_chunk_size, no_write,
           input_path, output_path):
    if not output_path:
        output_path = make_output_path(input_path)
    print('================ PARAMETERS')
    print(model_name, fps, random_seed, input_path, output_path, duration, )

    processor = InterpolationOfflineProcessor(model_name, fps, random_seed, frame_chunk_size)
    processor.process_file(input_path, output_path, start, duration, sr, not no_write, n_points)


@click.command()
@common_options
@click.option('--window_size', default=5, help='Window size', type=int)
@click.option('--displacement_factor', default=0.1, help='Displacement factor', type=float)
@click.option('--n_points', default=3, help='Number of points to interpolate between', type=int)
def spectro_interp(model_name, fps, random_seed, start, duration, sr, frame_chunk_size, no_write,
           input_path, output_path, window_size, displacement_factor, n_points):
    if not output_path:
        output_path = make_output_path(input_path)
    print('================ PARAMETERS')
    print(model_name, fps, random_seed, input_path, output_path, duration, )

    processor = SpectrogramInterpolationOfflineProcessor(model_name, fps, random_seed, frame_chunk_size)
    processor.process_file(input_path, output_path, start, duration, sr, not no_write, window_size, displacement_factor,
                           None, n_points)


cli.add_command(spectro)
cli.add_command(interp)
cli.add_command(spectro_interp)

if __name__ == '__main__':
    cli()
