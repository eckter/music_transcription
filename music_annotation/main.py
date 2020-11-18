from .training import train


def run():
    root = '/mnt/data/music'
    data = f'{root}/data'
    out = f'{root}/out'
    train(data, out)
