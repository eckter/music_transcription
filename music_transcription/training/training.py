import xml.etree.ElementTree as ET
import numpy as np
import torch
import matplotlib.pyplot as plt

from ..models import Network
from ..loader import get_loader, SongDataset
from ..models.network import samples_per_out
from ..loader.preprocess import freq_depth

from pathlib import Path


def is_valid(xml):
    xml = Path(xml)
    tree = ET.parse(str(xml))
    root = tree.getroot()
    tuning = root.find("tuning")
    tuning = tuning.attrib
    for string in tuning:
        if tuning[string] != "0":
            return False
    rhythm = Path(xml).parent / (xml.stem.split("_")[0] + "_rhythm.xml")
    if str(rhythm) == str(xml) or not rhythm.is_file():
        return True
    return is_valid(rhythm)


def get_mean(dataset):
    mean = np.zeros(freq_depth)
    for i in range(len(dataset)):
        x, _ = dataset[i]
        mean += x.mean(axis=0).numpy()
    mean /= len(dataset)
    return mean


def get_std(dataset, mean):
    std = np.zeros(freq_depth)
    for i in range(len(dataset)):
        x, _ = dataset[i]
        centered = x.numpy() - mean
        std += np.square(centered).mean(axis=0)
    std /= len(dataset)
    std = np.sqrt(std)
    return std


def loss_function(pred, target):
    loss = torch.nn.BCELoss(reduction="none")(pred, target)
    return loss.mean()


def eval_results(network, data, loss, device="cuda"):
    with torch.no_grad():
        network.eval()
        Xs = []
        ys = []
        for i in range(min(32, len(data))):
            X, y = data[i]
            Xs.append(X)
            ys.append(y)
        X = torch.stack(Xs)
        y = torch.stack(ys)
        pred, _ = network(X.to(device))
        y = torch.nn.MaxPool1d(samples_per_out)(y.transpose(1, 2).to(device)).transpose(1, 2)
        mask = y.max(dim=-1, keepdim=True)[0].expand_as(pred)
        pred = pred * mask
        return loss(pred, y).cpu().detach().item()


def range_to_1(l):
    return np.array(range(l)) / l


def plot_both(X, y, save_path=None):
    plt.figure(num=None, figsize=(8, 8), dpi=800, facecolor='w', edgecolor='k')

    from scipy import interpolate
    y = y.cpu().data.T
    if y.shape[0] == 1:
        y = np.repeat(y, 2, 0)
    X = X.cpu().data.T
    f = interpolate.interp2d(range_to_1(y.shape[1]), range_to_1(y.shape[0]), y)
    y = f(range_to_1(X.shape[1]), range_to_1(X.shape[0]))

    plt.imshow(X)
    plt.imshow(y, alpha=0.6, cmap="magma")
    plt.axis('off')
    plt.subplots_adjust(bottom=0, top=1, left=0, right=1)
    if save_path:
        plt.savefig(str(save_path))
    else:
        plt.show()


def train(data_root, out_root, workers=0):
    assert (Path(data_root).is_dir())
    Path(out_root).mkdir(exist_ok=True)

    root = Path(data_root)
    files = list(filter(is_valid, list(root.glob("*lead.xml"))[:-1]))
    val_files = files[:64]
    train_files = files[64:]

    train_data = SongDataset(train_files)
    val_data = SongDataset(val_files)

    train_data.sample_size = 2048
    val_data.sample_size = 2048

    if workers > 0:
        train_data.to_multi()
        train_data.multiplier = 40

    loader = get_loader(train_data, workers=workers, batch=32)

    mean = get_mean(train_data)
    std = get_std(train_data, mean)

    device = "cuda"
    network = Network(device, scaling_mean=mean, scaling_std=std).to(device)

    optimizer = torch.optim.Adam(network.parameters(), lr=2e-5)
    n_step = 0

    x_sample, y_sample = val_data[0]
    y_sample = torch.nn.MaxPool1d(samples_per_out)(y_sample.unsqueeze(0).transpose(1, 2)).transpose(1, 2)[0]
    plot_both(x_sample, y_sample, "base.png")

    for i in range(10000000):
        for X_train, y_train in loader:
            network.train()

            y_train = torch.nn.MaxPool1d(samples_per_out)(y_train.transpose(1, 2)).transpose(1, 2)
            y_train = y_train.to(device)
            x_train = X_train.to(device)

            optimizer.zero_grad()

            outputs, _ = network(x_train)
            tab_loss = loss_function(outputs, y_train)
            loss = tab_loss

            loss.backward()
            optimizer.step()

            if n_step % 200 == 0:
                train_loss = eval_results(network, train_data, loss_function, device=device)
                val_loss = eval_results(network, val_data, loss_function, device=device)
                print(f"{n_step}: train: {train_loss}, val: {val_loss}")
            if n_step % 2000 == 0:
                network.eval()
                output, _ = network(x_sample.unsqueeze(0).to(device))
                plot_both(x_sample, output, f"{n_step:09}.png")

            n_step += 1

        if n_step >= 100000:
            break

