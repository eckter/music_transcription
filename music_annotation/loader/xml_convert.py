from bisect import bisect_left
import numpy as np
import xml.etree.ElementTree as ET


def index_closest(l, x):
    if x > l[-1]:
        return -1
    i = bisect_left(l, x)
    if i > 0 and l[i] - x > x - l[i - 1]:
        return i
    else:
        return i + 1


def add_note(note, out, n_frets, sample_times):
    n = note.attrib
    time = float(n["time"])
    sustain = 0  # float(n.get("sustain", 0))
    end = time + sustain
    index = index_closest(sample_times, time) - 1
    index_end = index_closest(sample_times, end)
    if index < 0:
        return
    string = int(n["string"])
    fret = int(n["fret"])
    frame = np.zeros([6, n_frets + 1])
    frame[string][fret] = 1
    for i in range(index, index_end):
        out[i] += frame


def add_chord(c, chords, out, n_frets, sample_times):
    c = c.attrib
    time = float(c["time"])
    index = index_closest(sample_times, time) - 1
    if index < 0:
        return
    id = int(c["chordId"])
    chord = chords[id]

    frame = np.zeros([6, n_frets + 1])

    for string, fret in enumerate(chord):
        if fret >= 0:
            frame[string][fret] = 1
    out[index] += frame


def read_tab(xml, seq_length, sample_times, n_frets=25):
    tree = ET.parse(str(xml))
    root = tree.getroot()
    chords = list(root.find("chordTemplates"))
    chords = [[int(c.get(f"fret{i}")) for i in range(6)] for c in chords]
    levels = list(root.find("levels"))
    levels.sort(key=lambda x: int(x.get("difficulty")))

    out = np.zeros([seq_length, 6, n_frets + 1])

    for level in levels:
        notes = level.find("notes")
        chord_list = level.find("chords")

        for n in notes:
            add_note(n, out, n_frets, sample_times)
        if chord_list:
            for c in chord_list:
                add_chord(c, chords, out, n_frets, sample_times)
                for note in c:
                    add_note(note, out, n_frets, sample_times)

    for i in range(len(out)):
        for j in range(6):
            if out[i][j].mean() == 0:
                out[i][j][-1] = 1

    return out


def transform_one(y):
    y = np.array(y)
    string_differences = [0, 5, 10, 15, 19, 24]
    intermediate = np.zeros([y.shape[0], 25 + string_differences[-1]])
    for i, diff in enumerate(string_differences):
        intermediate[:, diff:diff+25] += y[:,i,:-1]

    return intermediate.clip(0, 1)


def read_and_transform(xml, seq_length, sample_times, n_frets=25):
    y = read_tab(xml, seq_length, sample_times, n_frets)
    return transform_one(y)

