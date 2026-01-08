# Setup

Install the libraries listed in `requirements.txt` or create a virtual enviroment using `setup_env.sh`. More information in `ENV_SETUP.md`.

# Training

1. Download europarl.en-ca.en.xz and europarl.en-ca.ca.xz from https://github.com/Softcatala/Europarl-catalan, unzip them and put it into a *data/* directory.

2. Run `src/nn_train.py

# Using the model

1. Download from https://huggingface.co/datasets/jlapeyra-upc/hle-project translator_seq2seq_v2.pt and save it in the root directory.

2. Run `src/nn_text.py`