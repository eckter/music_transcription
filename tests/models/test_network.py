from music_transcription.models.network import Network, samples_per_out
from music_transcription.loader.preprocess import freq_depth
from unittest.mock import patch, mock_open
import torch


def test_forward_shapes():
    """
    """
    network = Network(device="cpu")
    batch_size = 16
    length = 32
    data = torch.zeros(batch_size, length, freq_depth)
    out = network(data)
    expected_shape = torch.Size((batch_size, length // samples_per_out, 49))
    assert expected_shape == out.shape
