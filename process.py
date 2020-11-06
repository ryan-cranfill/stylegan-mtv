import click

from utils import make_output_path, download_audio_from_youtube
from src import add_src_to_sys_path, SpectrogramOfflineProcessor, InterpolationOfflineProcessor, \
    SpectrogramInterpolationOfflineProcessor

add_src_to_sys_path()


@click.group()
def cli():
    pass


def common_options(function):
    function = click.option('-m', '--model_name', default='wikiart-sg2', help='Model name (without .pkl extension)', type=str)(
        function)
    function = click.option('-f', '--fps', default=24, help='frames per second', type=int)(function)
    function = click.option('--random_seed', default=False, help='random seed', type=int)(function)
    function = click.option('-s', '--start', default=0, help='Start time', type=int)(function)
    function = click.option('-d', '--duration', default=None, help='Duration of video to make', type=int)(function)
    function = click.option('--sr', default=None, help='sample rate', type=int)(function)
    function = click.option('--frame_chunk_size', default=500, help='Number of frames to batch before writing to disk',
                            type=int)(function)
    function = click.option('--no_write', is_flag=True, help='Do not write out video.')(function)
    function = click.option('-o', '--output_path', default='', help='Output path', type=str)(function)
    function = click.option('-y', '--youtube_url', default=None, help='Youtube URl', type=str)(function)
    function = click.argument('input_path', nargs=-1)(function)
    return function


@click.command()
@common_options
@click.option('--window_size', default=5, help='Window size', type=int)
@click.option('--displacement_factor', default=0.1, help='Displacement factor', type=float)
def spectro(model_name, fps, random_seed, start, duration, sr, window_size, displacement_factor,
            frame_chunk_size, no_write, input_path, output_path, youtube_url):
    if youtube_url:
        input_path = download_audio_from_youtube(youtube_url)
    else:
        input_path = input_path[0]

    if not input_path:
        raise ValueError('Must provide input filepath or youtube path via -y')

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
@click.option('-l', '--likes_file', default=None, help='Path to likes file pickle', type=str)
def interp(model_name, n_points, fps, random_seed, start, duration, sr, frame_chunk_size, no_write,
           input_path, output_path, likes_file, youtube_url):
    if youtube_url:
        input_path = download_audio_from_youtube(youtube_url)
    else:
        input_path = input_path[0]

    if not input_path:
        raise ValueError('Must provide input filepath or youtube path via -y')

    if not output_path:
        output_path = make_output_path(input_path)
    print('================ PARAMETERS')
    print(model_name, fps, random_seed, input_path, output_path, duration, likes_file, n_points)
    processor = InterpolationOfflineProcessor(model_name, fps, random_seed, frame_chunk_size)
    processor.process_file(input_path, output_path, start, duration, sr, not no_write, n_points, likes_file)


@click.command()
@common_options
@click.option('--window_size', default=5, help='Window size', type=int)
@click.option('--displacement_factor', default=0.1, help='Displacement factor', type=float)
@click.option('--n_points', default=3, help='Number of points to interpolate between', type=int)
@click.option('-l', '--likes_file', default=None, help='Path to likes file pickle', type=str)
@click.option('--likes_dir', default=None, help='Directory with encoded .npy files to use as interpolation anchors', type=str)
@click.option('-c', '--config_file', default=None, help='Path to configuration json file', type=str)
def spectro_interp(model_name, fps, random_seed, start, duration, sr, frame_chunk_size, no_write,
                   input_path, output_path, window_size, displacement_factor, n_points, likes_file, likes_dir,
                   youtube_url, config_file):
    if youtube_url:
        input_path = download_audio_from_youtube(youtube_url)
    else:
        input_path = input_path[0]

    if not input_path:
        raise ValueError('Must provide input filepath or youtube path via -y')

    if not output_path:
        output_path = make_output_path(input_path)

    print('================ PARAMETERS')
    print(model_name, fps, random_seed, input_path, output_path, duration,)

    processor = SpectrogramInterpolationOfflineProcessor(model_name, fps, random_seed, frame_chunk_size)
    processor.process_file(input_path, output_path, start, duration, sr, not no_write, window_size, displacement_factor,
                           None, n_points, likes_file, config_file, likes_dir)


cli.add_command(spectro)
cli.add_command(interp)
cli.add_command(spectro_interp)

if __name__ == '__main__':
    cli()
