from pathlib import Path
from scipy import interpolate
import librosa
import numpy as np
import soundfile
import xml.etree.ElementTree as ET
from multiprocessing import Array
from kymatio.numpy import Scattering1D

from .xml_convert import read_and_transform

freq_depth = 128


def get_harmonics(data):
    x = librosa.stft(data)
    h, p = librosa.decompose.hpss(x)
    return librosa.istft(h)


def resample(S, samples, music_time):
    f = np.arange(S.shape[0])

    times = np.arange(S.shape[1]) / S.shape[1] * music_time

    func = interpolate.interp2d(times, f, S)

    new_times = samples

    resampled = func(new_times, f)
    return resampled


def read_spectro_samples(ogg, samples):
    data, samplerate = soundfile.read(str(ogg), always_2d=True)
    data = data.mean(axis=1)

    #data = get_harmonics(data)
    data = np.concatenate([data, np.zeros(int(samplerate * 0.5))])
    music_time = len(data) / samplerate

    mfcc = librosa.feature.mfcc(y=data, sr=samplerate, n_mfcc=64 - 12)
    chroma = librosa.feature.chroma_cqt(y=data, sr=samplerate)
    # spectro = librosa.feature.melspectrogram(y=data, sr=samplerate, n_mels=128, n_fft=2048)
    spectro = librosa.stft(data, n_fft=126)
    spectro_magnitude = librosa.power_to_db(np.abs(spectro))

    T = data.shape[-1]
    J = 7
    Q = 16
    #  scattering = Scattering1D(J, T, Q)(data)
    #  scattering = scattering[:128, :]

    mfcc = resample(mfcc, samples, music_time)
    chroma = resample(chroma, samples, music_time)
    spectro = resample(spectro_magnitude, samples, music_time)
    #  scattering = resample(scattering, samples, music_time)

    res = np.concatenate([mfcc, chroma, spectro], axis=0).T
    assert res.shape[1] == freq_depth, f"{res.shape[1]} != {freq_depth}"
    return res


def get_sample_times(xml, samples_per_out=4):
    tree = ET.parse(str(xml))
    root = tree.getroot()
    beats = [float(x.get("time")) for x in root.find("ebeats")]
    Xs = np.array(range(len(beats)))
    f = interpolate.interp1d(Xs, beats)

    sample_per_beat = 8 * samples_per_out

    new_Xs = np.array(range((len(beats) - 1) * sample_per_beat)) / sample_per_beat
    return f(new_Xs)


def load_audio(f):
    f = Path(f)
    sample_times = get_sample_times(f)
    name = f.stem.split("_")[0]
    ogg = Path(f).parent / "output2" / name / "other.wav"
    if not ogg.is_file():
        ogg = Path(f).parent / (name + ".ogg")
    assert(ogg.is_file())
    audio_processed = read_spectro_samples(ogg, sample_times).reshape(-1)
    return audio_processed

def load_tab(f):
    f = Path(f)
    sample_times = get_sample_times(f)
    name = f.stem.split("_")[0]
    t = read_and_transform(f, sample_times)
    rhythm = Path(f).parent / (name + "_rhythm.xml")
    if rhythm.is_file():
        t += read_and_transform(rhythm, sample_times)
        t = t.clip(0, 1)
    t = t.reshape(-1)
    return t
