from .training import train


def run():
    root = '.'
    data = f'{root}/data'
    out = f'{root}/out'
    train(data, out)
