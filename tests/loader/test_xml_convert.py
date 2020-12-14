from music_transcription.loader.xml_convert import read_and_transform
import numpy as np
from pathlib import Path




def test_read_and_transform__note_sync():
    xml = Path(__file__).parent / "anotations_file_lead.xml"
    first_note_time_expected = 11.3
    last_note_time_expected = 369
    delta = 0.02
    sample_times = np.arange(0, 6 * 60 + 15, delta)
    res = read_and_transform(xml, sample_times)
    res = res.max(axis=-1)
    first_note = np.argmax(res) * delta
    last_note = (len(res) - 1 - np.argmax(res[::-1])) * delta

    assert abs(first_note - first_note_time_expected) < 0.1
    assert abs(last_note - last_note_time_expected) < 0.1
