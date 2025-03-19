## Configuration

1. Create the environment, activate it and install the content of `requirements.txt`
2. Download the NLTK data: `python -c "import nltk; nltk.download('punkt_tab')"`
3. Clone StyleTTS2 ( https://github.com/yl4579/StyleTTS2 ) in its directory (let's name it `STYLE_TTS2_ROOT`)
4. Go to the StyleTTS directory and run `git-lfs clone https://huggingface.co/yl4579/StyleTTS2-LibriTTS\nmv StyleTTS2-LibriTTS/Models .`
2. Come back to the current directory, and set STYLETTS_PATH accordingly in styletts.py file
5. Everything is ready to be run using the command `python gradio_gui.py`
