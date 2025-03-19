import gradio as gr

import embedding_control_styletts as ctrl

import os, tempfile






public=False

tempdir = tempfile.gettempdir()
system = "styletts2"

#system = "openvoice"
#system = "xtts"

tts = ctrl.StyleControls(system)
phon_controls  = {item: item for item in tts.feats}
style_control = True
styles = {item: item for item in tts.styles}

def speak(text, *args):

    print(args)
    if style_control:
        (style, style_mul, ref_file, orth_f, *args) = args
    else:
        (ref_file, orth_f, *args) = args
    #style = "neutral"
    text = text.lower()
   
    spkr_embedding  = tts.get_embeddings(ref_file.name)
    
    if style_control and style != "default":
        spkr_embedding_mod = tts.mod_style('style', style)
        spkr_embedding += style_mul * spkr_embedding_mod

    for i, feat in enumerate(tts.feats):
       
        try:
            val = args[i]
            spkr_embedding  = tts.mod_embeddings(feat, val, orth_f, spkr_embedding)
    
        except:
            print(key+" failed.")
            pass

    audio = tts.speak(text, spkr_embedding, tempdir+"/tmp.wav")
    
    if not public:
        try:
            pass
            os.system("play -q "+tempdir+"/tmp.wav &")
        except:
            pass

    return (24000, audio)



controls = []
controls.append(gr.Textbox(label="text", value="A quick brown fox jumped over a lazy old dog."))
if style_control: 
    controls.append(gr.Dropdown(styles, label="style"))
    controls.append(gr.Slider(minimum=-1, maximum=2,step=0.05, value=0, label="emotion strength"))

controls.append(gr.UploadButton(file_types=[".wav"], label="Upload reference audio"))
controls.append(gr.Checkbox(label="orthogonalize", value=False))

for feat in tts.feats:
    controls.append(gr.Slider(minimum=-2, maximum=2, step=0.05, value=0, label=feat))



tts_gui = gr.Interface(
    fn=speak,
    inputs=controls,
    outputs= gr.Audio(label="output"),
    live=False

)


if __name__ == "__main__":
    tts_gui.launch(share=public)
