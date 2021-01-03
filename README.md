# stylegan-mtv
Use a StyleGAN model to generate videos reacting to sound.

# Spectrogram Only
```
python process.py spectro [OPTIONS] INPUT_PATH OUTPUT_PATH

Options:
  --model_name TEXT            model name
  --fps INTEGER                frames per second
  --random_seed INTEGER        random seed
  --start INTEGER              Start time
  --duration INTEGER           Duration of video to make
  --sr INTEGER                 sample rate
  --window_size INTEGER        Window size
  --displacement_factor FLOAT  Displacement factor
  --no_write                   Do not write out video.
  --help                       Show help and exit.
```

# Docker Version
- Get docker running (if on windows, [this guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#setting-containers)
  is helpful)
- create a `models` directory and put your stylegan models in there
- `docker build -t mtv .` to build container (will take a few minutes to download and build everything)

- start a dev machine that gets deleted when you're done with a command like 
  `docker run --gpus all --shm-size=4g --ulimit memlock=-1 --ulimit stack=97108864 -v $PWD:/code -it --rm mtv bash` 
  or don't use `--rm` if you want it to persist
- test with something like `python process.py spectro -y https://www.youtube.com/watch?v=_V2sBURgUBI -d 15 --window_size 30`
