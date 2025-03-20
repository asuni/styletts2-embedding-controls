#!/bin/bash

# Install pre-requisites
echo "==== Install dependencies (including nltk database)"
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab')"

# Retrieve StyleTTS if necessary
echo "==== Retrieve StyleTTS2 (only if necessary)"
if [ ! -f $PWD/StyleTTS2/README.md ]; then
    git submodule update --init
fi

# Retrieve LibriTTS models if necessary
echo "==== Retrieve StyleTTS2 LibriTTS models (only if necessary)"
if [ ! -f $PWD/StyleTTS2/Models/LibriTTS/config.yaml ]; then
    (
        cd StyleTTS2;
        git clone https://huggingface.co/yl4579/StyleTTS2-LibriTTS
        mv StyleTTS2-LibriTTS/Models .
    )
fi
