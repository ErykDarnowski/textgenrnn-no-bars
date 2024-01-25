import os
import tensorflow as tf
from textgenrnn import textgenrnn

os.system('clear')

if not os.path.exists("./textgenrnn_weights.hdf5"):
    tf.keras.utils.enable_interactive_logging()

    # training new model
    generator = textgenrnn()
    generator.train_from_file("clean.txt",
        header=False,
        new_model=True,
        num_epochs=300)
else:
    tf.keras.utils.disable_interactive_logging()

    # loading from trained model
    generator = textgenrnn(
        weights_path="textgenrnn_weights.hdf5",
        vocab_path="textgenrnn_vocab.json",
        config_path="textgenrnn_config.json")

    #generator.generate(20, temperature=1.9)
    generator.generate_samples(5)
