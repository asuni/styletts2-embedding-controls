import gradio as gr

import embedding_control_styletts as ctrl




import os, tempfile

import numpy as np




public=False

tempdir = tempfile.gettempdir()

tts = ctrl.StyleControls() 
phon_controls  = {item: item for item in tts.feats}

styles = {item: item for item in tts.styles}
print(tts.feats)
def speak(text, *args):

    print(args)

    (style, style_mul, style2, style_mul2, method, ref_file, orth_f, *args) = args
    #else:
    #    (ref_file, orth_f, *args) = args
    #style = "neutral"
    text = text.lower()
   
    spkr_embedding  = tts.get_embeddings(ref_file.name)
    orig_embedding = np.array(spkr_embedding)
    
    
    if method =="centroid":
        if style != "default":
            style_vec = tts.get_style_vector('style', style, use_median=orth_f)
            spkr_embedding += style_mul * style_vec
        if style2 != "default":
            style_vec = tts.get_style_vector('style', style2, use_median=orth_f)
            spkr_embedding += style_mul2 * style_vec


    elif method == "svm":
        if style != "default":     
            style_vec, alpha = tts.get_svm_style_vector(style)
            spkr_embedding= spkr_embedding + style_vec*alpha*style_mul
        if style2 != "default":
            style_vec, alpha = tts.get_svm_style_vector(style2)
            spkr_embedding= spkr_embedding + style_vec*alpha*style_mul2
        
    #"""
    for i, feat in enumerate(tts.feats):
       
        try:
            val = args[i]
            spkr_embedding  = tts.add_prosody_vector(feat, val, False, spkr_embedding)

        except:
            print(key+" failed.")
            pass
    #"""
    #spkr_embedding[:128] = orig_embedding[:128]
    audio = tts.speak(text, spkr_embedding, tempdir+"/tmp.wav") # ,alpha=alpha, beta=beta, emb_scale=emb_scale)
    
    if not public:
        try:
            pass
            os.system("play -q "+tempdir+"/tmp.wav &")
        except:
            pass

    return [ref_file.name, (24000, audio)]



controls = []
controls.append(gr.Textbox(label="text", value="A quick brown fox jumped over a lazy old dog."))
if 1==1: 
    controls.append(gr.Dropdown(styles, label="style", value="default"))
    controls.append(gr.Slider(minimum=-1, maximum=2,step=0.05, value=1, label="style strength"))
    controls.append(gr.Dropdown(styles, label="style2", value="default"))
    controls.append(gr.Slider(minimum=-1, maximum=2,step=0.05, value=0, label="style2 strength"))
   

    controls.append(gr.Radio(["centroid", "svm"], label="method", value="centroid"))

   
controls.append(gr.UploadButton(file_types=[".wav"], label="Upload reference audio"))
controls.append(gr.Checkbox(label="orthogonalize", value=False))

for feat in tts.feats:
    controls.append(gr.Slider(minimum=-3, maximum=3, step=0.1, value=0, label=feat))



tts_gui = gr.Interface(
    fn=speak,
    inputs=controls,
    outputs= [
        gr.Audio(label="reference", type="filepath"),
        gr.Audio(label="output")
    ],
    live=False

)


if __name__ == "__main__":
    tts_gui.launch(share=public)
