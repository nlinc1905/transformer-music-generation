import os
from pathlib import Path
import random
import pandas as pd
import numpy as np
from music21 import converter, instrument, note, chord, stream
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


class Processor:
    def __init__(self, seed, vocab_size, max_sequence_len, batch_size):
        self.seed = seed
        self.vocab_size = vocab_size
        self.max_sequence_len = max_sequence_len
        self.batch_size = batch_size
        # placeholders for future attributes
        self.text_ds, self.vocab = (None, None)

    @staticmethod
    def offsets_relative_to_prior_note(offsets):
        """
        Converts a list of offsets relative to the start of a song, 0, to a list relative
        to the time since each previous note.
        Example:
            [0, 2, 3, 5]  -->  [0, 2, 1, 2]
        """
        return [v - offsets[i - 1] if i > 0 else 0 for i, v in enumerate(offsets)]

    @staticmethod
    def offsets_relative_to_beginning(offsets):
        """
        Converts a list of offsets relative to the time since each previous note to a list relative
        to the start of a song, 0.
        Example:
            [0, 2, 1, 2]  -->  [0, 2, 3, 5]
        """
        return np.cumsum(offsets).tolist()

    @staticmethod
    def collapse_chord_name_to_string(c):
        """
        Chord name strings look like: '<music21.chord.Chord B4 B6>'
        This function extracts the note names from the end and joins them with '.'
        so the chord comes out looking like: 'B4.B6'

        :param c: a music21.chord.Chord object
        """
        c = str(c)
        note_names_start_after = "Chord "
        note_string = c[c.index(note_names_start_after)+len(note_names_start_after):c.rindex(">")]
        return note_string.replace(' ', '.')

    @staticmethod
    def parse_midi_file(file_path):
        """
        Reads a MIDI files and parses it to a sequence of notes and chords.
        Note times are discarded, so there may be some notes and chords that are played
        simultaneously that end up following the other in the sequence.  But the goal
        is to end up with a 1 dimensional sequence of notes.

        Ignore rests (isinstance(n, note.Rest), with n.name property)

        :return: list of note/chord strings, and a list of offsets relative to the start of the song
        """
        midi = converter.parse(file_path)
        stream_score = instrument.partitionByInstrument(midi)
        notes_to_parse = stream_score.parts[0].recurse()
        notes, offsets = [], []
        for n in notes_to_parse:
            # append note
            if isinstance(n, note.Note):
                notes.append(str(n.pitch))
                # add offset to set, default to float
                if str(type(n.offset)) == "<class 'fractions.Fraction'>":
                    offsets.append((n.offset.numerator / n.offset.denominator))
                else:
                    offsets.append(n.offset)
            # if chord, append a collapsed string to represent all notes in chord
            elif isinstance(n, chord.Chord):
                notes.append(Processor.collapse_chord_name_to_string(n))
                # add offset to set, default to float
                if str(type(n.offset)) == "<class 'fractions.Fraction'>":
                    offsets.append((n.offset.numerator / n.offset.denominator))
                else:
                    offsets.append(n.offset)

        return notes, offsets

    @staticmethod
    def sequence_to_midi(notes_list, offsets=None):
        """
        Converts a sequence of notes/chords to a MIDI file

        :param notes_list: list of note names like ['A', 'C#', 'B-', 'C.E']
        :param offsets: list of note offset times like [0, 3, 5.2], relative to the start of the song

        :return: MIDI stream of notes
        """
        # if no offsets provided, equally space each note by 0.5
        if offsets is None:
            offsets = [i for i in np.arange(0, len(notes_list), 0.5)]
        output_notes = []
        for pattern_idx, pattern in enumerate(notes_list):
            # parse chords
            if ('.' in pattern):
                notes_in_chord = pattern.split('.')
                notes = []
                # append each note in the chord to a piano instrument track
                for current_note in notes_in_chord:
                    new_note = note.Note(current_note)
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                # create a chord from the notes in the list
                new_chord = chord.Chord(notes)
                # offset the chord timing by the current offset amount
                new_chord.offset = offsets[pattern_idx]
                output_notes.append(new_chord)
            # parse individual note
            else:
                new_note = note.Note(pattern)
                # set the note's instrument property to piano
                new_note.storedInstrument = instrument.Piano()
                # offset the note timing by the current offset amount
                new_note.offset = offsets[pattern_idx]
                output_notes.append(new_note)
        # stream the output to MIDI
        return stream.Stream(output_notes)

    def prepare_model_input(self):
        """
        Reads and parses MIDI files, then saves the note sequences as text files in the
        train and val directories.

        Note that each song is treated as 1 long sentence, which has implications for the
        max_sequence_length argument passed to the transformer.  In the future, it may be worth
        exploring how to break up the notes into sentence-like chunks, perhaps by using bars
        like MuseGAN.

        :return: TF TextLineDataset object with x and y mapped.
        """
        # if train and val directories do not exist, create them
        cwd = os.getcwd()
        data_dir = cwd + "/maestro-v3.0.0-midi/maestro-v3.0.0/"
        Path(cwd + "/clean/train").mkdir(parents=True, exist_ok=True)
        Path(cwd + "/clean/val").mkdir(parents=True, exist_ok=True)

        # read Maestro metadata to Pandas, shuffle, then determine train/test split
        metadata = pd.read_csv(data_dir + "maestro-v3.0.0.csv")
        metadata = metadata.sample(frac=1.0)
        train_val_split_index = int(len(metadata) * 0.8)

        # only parse MIDI files if the clean train or test dirs are empty
        if len(os.listdir('clean/train')) == 0 or len(os.listdir('clean/val')) == 0:
            # use the Pandas df to determine which parsed files to store in each folder
            for file_idx, file in enumerate(metadata.midi_filename.to_list()):
                # Pre-pend the data directory to the file path
                file = data_dir + file
                print("Parsing MIDI file:", file_idx, "/", len(metadata))
                notes_list, offsets = self.parse_midi_file(file)
                offsets_from_prior_note = self.offsets_relative_to_prior_note(offsets=offsets)
                if file_idx <= train_val_split_index:
                    path_prefix = "clean/train/"
                else:
                    path_prefix = "clean/val/"
                # save each parsed sequence of notes as a string in a txt file
                with open(path_prefix + file[file.rindex("/")+1:file.rindex(".")] + ".txt", "w") as text_file:
                    text_file.write(' '.join(notes_list))

        # walk through the directories
        filenames = []
        directories = [
            "clean/train",
            "clean/val",
        ]
        for dir in directories:
            for f in os.listdir(dir):
                filenames.append(os.path.join(dir, f))

        print(f"{len(filenames)} files")
        
        # Create a dataset from text files
        random.shuffle(filenames)
        text_ds = tf.data.TextLineDataset(filenames)
        text_ds = text_ds.shuffle(buffer_size=256)
        text_ds = text_ds.batch(self.batch_size)

        # Create a vectorization layer and adapt it to the text
        vectorize_layer = TextVectorization(
            standardize=None,  # do not perform any pre-processing or cleaning
            max_tokens=self.vocab_size - 1,
            output_mode="int",
            output_sequence_length=self.max_sequence_len + 1,
        )
        vectorize_layer.adapt(text_ds)
        self.vocab = vectorize_layer.get_vocabulary()

        def create_x_and_y(text):
            """
            Shift word sequences by 1 position so that the target for position (i) is
            word at position (i+1). The model will use all words up till position (i)
            to predict the next word.
            """
            text = tf.expand_dims(text, -1)
            tokenized_sentences = vectorize_layer(text)
            x = tokenized_sentences[:, :-1]
            y = tokenized_sentences[:, 1:]

            return x, y

        return text_ds.map(create_x_and_y)


# Comments below helpful for debugging and running parts of this script in isolation

# p = Processor(seed=14, vocab_size=20000, max_sequence_len=100, batch_size=128)
# notes_list, offsets = p.parse_midi_file(
#     'maestro-v3.0.0-midi/maestro-v3.0.0/2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.midi'
# )
# offsets_from_prior_note = p.offsets_relative_to_prior_note(offsets=offsets)

# midi_stream = Processor.sequence_to_midi(notes_list=notes_list, offsets=offsets)
# midi_stream = Processor.sequence_to_midi(notes_list=notes_list)
# midi_stream.write('midi', fp='generated_midi_outputs/test_output.mid')

# p = Processor(seed=14, vocab_size=20000, max_sequence_len=100, batch_size=128)
# output = p.prepare_model_input()
