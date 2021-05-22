import argparse
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from data_processor import Processor
from models import DeepTransformer


def execute(starting_note, seq_len):
    CONFIG = {
        'seed': 14,
        'max_sequence_length': 200,  # max tokens - note that each song is treated as 1 sequence
        'vocabulary_size': 20000,
        'learning_rate': 1e-3,
        'epochs': 6,
        'batch_size': 128,
        'transformer_embed_dim': 64,  # 256,
        'transformer_blocks': 1,
        'transformer_heads': 2,
        'transformer_units': 64,  # 256,
    }
    with open("model_tracking/models.config", "w") as file:
        json.dump(CONFIG, file)

    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    tf.random.set_seed(CONFIG['seed'])

    p = Processor(
        seed=CONFIG['seed'],
        vocab_size=CONFIG['vocabulary_size'],
        max_sequence_len=CONFIG['max_sequence_length'],
        batch_size=CONFIG['batch_size'],
    )
    map_dataset = p.prepare_model_input()

    trns = DeepTransformer(seed=CONFIG['seed'], checkpoint_path="model_tracking/trns.ckpt")
    trns.build(
        max_seq_len=CONFIG['max_sequence_length'],
        vocab_size=CONFIG['vocabulary_size'],
        # embed_dim determines units in dense layer for input values, and dims for positional embeddings
        embed_dim=CONFIG['transformer_embed_dim'],
        loss=SparseCategoricalCrossentropy(from_logits=True),
        learning_rate=CONFIG['learning_rate'],
        nbr_transformer_blocks=CONFIG['transformer_blocks'],
        nbr_attention_heads_each_block=CONFIG['transformer_heads'],
        nbr_dense_units_each_block=CONFIG['transformer_units'],
    )
    trns.train(map_dataset=map_dataset, epochs=CONFIG['epochs'])

    # make predictions, given a starting note
    generated_notes_list = trns.make_prediction_from_prompt(
        start_prompt=starting_note,
        text_processor=p,
        prediction_len=seq_len
    )
    generated_midi_seq = p.sequence_to_midi(
        notes_list=generated_notes_list,
        offsets=None
    )
    generated_midi_seq.write(
        'midi',
        fp='generated_midi_outputs/generated_output.mid'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--start_note',
        help="First note in the sequence to generate, like 'A5'"
    )
    parser.add_argument(
        '--seq_len',
        type=int,
        help="How many notes to generate."
    )
    args = parser.parse_args()
    execute(starting_note=args.start_note, seq_len=args.seq_len)


"""
Idea for improvement...

The model could be improved just by adding the offset data.  Right now, no offsets are passed to the MIDI generator, 
so a default of 0.5 is used.  The song would sound more musical if it could make use of the offset data.  Once way 
might be to feed the notes to an embedding & do positional embedding, just like a regular language transformer.  
Then feed the offsets to a dense layer, which could be concatenated with the output of the transformer, before it is 
passed to the final dense layer.  This link might help:
https://stackoverflow.com/questions/51360827/how-to-combine-numerical-and-categorical-values-in-a-vector-as-input-for-lstm
"""

