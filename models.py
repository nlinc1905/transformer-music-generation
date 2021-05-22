import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense,
    Layer, Dropout, MultiHeadAttention, LayerNormalization,
    Embedding, GlobalAveragePooling2D, GlobalAveragePooling1D
)
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    A causal mask forces predictions to only attend to the tokens at previous positions, so that the model
    is autoregressive.  Essentially, the mask hides future tokens/values, or tokens to the right of the
    current token, when attention is applied.

    This causal attention mask is implemented as a boolean tensor, with the upper half of the dot product
    matrix masked.

    Example mask when n_dest = n_src = 5:
        <tf.Tensor: shape=(5, 5), dtype=bool, numpy=
        array([[ True,  True,  True,  True,  True],
               [False,  True,  True,  True,  True],
               [False, False,  True,  True,  True],
               [False, False, False,  True,  True],
               [False, False, False, False,  True]])>
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    # create the mask as a boolean tensor
    mask = i >= j - n_src + n_dest
    mask = tf.cast(mask, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    # create a tensor to help tile the mask so there's 1 mask per sample in the batch
    # mult should look like: [batch_size, 1, 1]
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    # return a tensor mask of shape [batch_size, n_dest, n_src]
    return tf.tile(mask, mult)


class TransformerBlock(Layer):
    """
    Class borrowed from: https://keras.io/examples/nlp/text_classification_with_transformer/
    Note: this is only the decoder block from a full transformer, and it omits the second attention
    layer so that it matches the GPT architectures, see: http://jalammar.github.io/illustrated-gpt2/
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim), ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size=batch_size, n_dest=seq_len, n_src=seq_len, dtype=tf.bool)
        attn_output = self.att(query=inputs, value=inputs, attention_mask=causal_mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(Layer):
    """
    These 2 sources helped inspire this class:
        https://keras.io/examples/nlp/text_classification_with_transformer/
        https://keras.io/examples/vision/image_classification_with_vision_transformer/
    """
    def __init__(self, sequence_len, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=sequence_len, output_dim=embed_dim)

    def call(self, x):
        sequence_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=sequence_len, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class DeepTransformer:
    def __init__(self, seed, checkpoint_path):
        self.seed = seed
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.training_hist = None

    def build(self, max_seq_len, vocab_size, embed_dim, loss, learning_rate,
              nbr_transformer_blocks=1, nbr_attention_heads_each_block=2, nbr_dense_units_each_block=32):
        """
        Builds the transformer architecture.  For advice in choosing the number of attention heads, see:
            https://blog.ml.cmu.edu/2020/03/20/are-sixteen-heads-really-better-than-one/
        Spoiler: more heads 'can' help with training, but you are likely better off using as few as necessary

        :param max_seq_len: Only the first max_seq_len items/words/time_steps in each sequence will be modeled.  This
            is equivalent to nbr_time_steps in the other models in this script, as the max_seq_len will simply be the
            number of total time steps.
        :param vocab_size: Maximum number of unique tokens in the vocabulary
        :param embed_dim: The number of latent features/embedding dimensions for the token and position embeddings.
            Setting this equal to nbr_features will ignore dimension reduction and not compress anything.
        :param loss: Type of loss to optimize
        :param learning_rate: Controls the size of the adjustments to the model's weights in each iteration
        :param nbr_transformer_blocks: How deep to make the network
        :param nbr_attention_heads_each_block: How many heads in the multi-headed attention unit, for each transformer
        :param nbr_dense_units_each_block: How many units in the dense/feedforward part of each transformer

        :return: None
        """
        i = Input(shape=(max_seq_len,), dtype=tf.int32)
        x = TokenAndPositionEmbedding(sequence_len=max_seq_len, vocab_size=vocab_size, embed_dim=embed_dim)(i)
        for layer in range(nbr_transformer_blocks):
            x = TransformerBlock(embed_dim, nbr_attention_heads_each_block, nbr_dense_units_each_block)(x)
        outputs = Dense(vocab_size)(x)
        self.model = Model(inputs=i, outputs=[outputs, x])
        self.model.compile(
            loss=[loss, None],
            optimizer=Adam(lr=learning_rate),
        )
        print(self.model.summary())

    def load_saved_weights(self):
        self.model.load_weights(self.checkpoint_path)

    def train(self, map_dataset, epochs):
        # callback to monitor training and save weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            save_weights_only=True,
            verbose=1
        )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.checkpoint_path, histogram_freq=1
        )

        self.training_hist = self.model.fit(
            map_dataset,
            epochs=epochs,
            callbacks=[cp_callback, tensorboard_callback],
        )
        return self.training_hist

    def _sample_from_distribution(self, logits, top_k):
        logits, indices = tf.math.top_k(logits, k=top_k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def _detokenize(self, text_processor, number):
        return text_processor.vocab[number]

    def make_prediction_from_prompt(self, start_prompt, text_processor, prediction_len=50):
        """
        Given a starting note (start_prompt), make recursive one-step predictions up to the
        given prediction_len.
        """
        # tokenize the starting prompt using the given vocabulary
        word_to_index = {word:index for index, word in enumerate(text_processor.vocab)}
        start_tokens_original = [word_to_index.get(_, 1) for _ in start_prompt.split()]
        start_tokens = [_ for _ in start_tokens_original]

        # iteratively predict 1 step ahead to generate a sequence
        nbr_tokens_generated = 0
        tokens_generated = []
        while nbr_tokens_generated <= prediction_len:

            # pad the input prompt to match the shape the model was trained on
            pad_len = text_processor.max_sequence_len - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:text_processor.max_sequence_len]
                sample_index = text_processor.max_sequence_len - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])

            # make prediction on given x
            y, _ = self.model.predict(x)

            # generate a sample from the probability distribution of next tokens,
            # then append it to the prompt for the next iteration
            sample_token = self._sample_from_distribution(logits=y[0][sample_index], top_k=10)
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            nbr_tokens_generated = len(tokens_generated)

        # combine the de-tokenized text to a string
        text_generated = " ".join(
            [self._detokenize(text_processor=text_processor, number=n)
             for n in start_tokens_original + tokens_generated]
        )

        # drop notes outside the vocabulary and convert to a list
        text_generated = text_generated.replace("[UNK]", "").strip().split(" ")
        # drop empty strings
        text_generated = [t for t in text_generated if t != ""]

        return text_generated
