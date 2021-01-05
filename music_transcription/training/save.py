import torch


def save(path, network, optimizer, step, **kwargs):
    torch.save({
            "network": network,
            "optimizer": optimizer,
            "step": step,
            **kwargs
        },
        path)


def load_as_dict(path):
    return torch.load(path)


def load(path):
    d = load_as_dict(path)
    return d["network"], d["optimizer"], d["step"]
