from pathlib import Path
import pickle
from .preprocess import load_audio, load_tab, freq_depth

save_root = Path(__file__).parents[2] / "preprocessed_data"
version_audio = "1.0"
version_tab = "1.0"


def try_read_file(save_path, backup_func):
    try:
        return pickle.load(open(save_path, "rb"))
    except:
        data = backup_func()
        p = pickle.Pickler(open(str(save_path), "wb"))
        p.fast = True
        p.dump(data)
        return data


def load_with_saves(xml_path, multithread=False):
    file_id = xml_path.stem
    save_file_name = f"{file_id}.p"

    audio_path = save_root / "audio" / version_audio / save_file_name
    tab_path = save_root / "tabs" / version_tab / save_file_name

    for p in [audio_path, tab_path]:
        p.parent.mkdir(parents=True, exist_ok=True)
    
    a = try_read_file(audio_path, lambda: load_audio(xml_path))
    t = try_read_file(tab_path, lambda: load_tab(xml_path, a.shape[0] // freq_depth))

    if multithread:
        a = Array('f', s, lock=False)
        t = Array('f', t, lock=False)
    return a, t
