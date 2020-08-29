import os
import pafy
import string
import youtube_dl
from pathlib import Path

from src.settings import OUTPUT_DIR, INPUT_DIR


def format_filename(s):
    """Take a string and return a valid filename constructed from the string.
Uses a whitelist approach: any characters not present in valid_chars are
removed. Also spaces are replaced with underscores.

Note: this method may produce invalid filenames such as ``, `.` or `..`
When I use this method I prepend a date string like '2009_01_15_19_46_32_'
and append a file extension like '.txt', so I avoid the potential of using
an invalid filename.

"""
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ', '_')  # I don't like spaces in filenames.
    return filename


def uniquify(path: [str, Path]):
    if isinstance(path, Path):
        path = str(path)

    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


def make_output_path(input_path: [str, Path], out_dir=OUTPUT_DIR):
    if isinstance(input_path, str):
        input_path = Path(input_path)
    if not isinstance(out_dir, Path):
        out_dir = Path(out_dir)

    out_path = out_dir / input_path.with_suffix('.mp4').parts[-1]
    return uniquify(out_path)


def clear_ydl_cache():
    print('clearing ydl cache')
    with youtube_dl.YoutubeDL({}) as ydl:
        ydl.cache.remove()


def download_audio_from_youtube(url):
    clear_ydl_cache()

    print('getting video...')
    video = pafy.new(url)
    print('video got...')
    title = video.title
    bestaudio = video.getbestaudio()
    filename = format_filename(f'{title}.{bestaudio.extension}')
    path = INPUT_DIR / filename

    if not path.exists():
        bestaudio.download(str(path))
    else:
        print(path, 'exists, not downloading!')

    return path


    # print(url)
    # ydl_opts = {
    #     'format': 'bestaudio',
    #     'postprocessors': [{
    #         'key': 'FFmpegExtractAudio',
    #         'preferredcodec': 'mp3',
    #         'preferredquality': '192',
    #     }],
    #     'outtmpl': str(INPUT_DIR) + '/%(title)s.%(etx)s',
    #     'quiet': False
    # }
    #
    #
    #
    # with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    #     info = ydl.extract_info(url, download=False)
    #     print(info)
    #
    #     filename = ydl.prepare_filename(info)
    #     path = Path(filename)
    #     # info = ydl.extract_info(url, download=False)
    #     # # print(info)
    #     # path = str(INPUT_DIR / info['title'] / '.mp3')
    #     # ydl.download([url])
    #     return path.with_suffix('.mp3')
