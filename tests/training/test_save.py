from music_transcription.models.network import Network
from music_transcription.loader.preprocess import freq_depth
from music_transcription.training.save import load, save
import torch
import os


def test_load_save():
    """
    """
    network = Network(device="cpu")
    optimizer = torch.optim.Adam(network.parameters(), lr=4242)
    step = 12345
    path = "tmp"
    save("tmp", network, optimizer, step)
    network2, optimizer2, step2 = load(path)
    os.remove(path)

    batch_size = 16
    length = 32
    data = torch.zeros(batch_size, length, freq_depth)
    with torch.no_grad():
        network.eval()
        network2.eval()
        out1 = network(data)
        out2 = network2(data)
        assert torch.equal(out1, out2)

    assert step2 == step

