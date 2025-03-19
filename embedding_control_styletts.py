import os, sys, glob
import numpy as np
import time
import torch
import torchaudio


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import parselmouth
from tqdm import tqdm


class StyleControls():
    def __init__(self, synthesizer="styletts2",prosody_dir=None, style_dir=None):
        self.device = "cuda"
        self.synthesizer = synthesizer
        if self.synthesizer == "styletts2":
            import styletts
            self.model = styletts.StyleTTS()
        else:
            print("Synthesizer "+synthesizer+ "not supported.")
            sys.exit(0)

        self.feats = []
        self.feat_std = {}    
        self.styles = []

        self.control_coeffs = {}
        self.control_weights = {}

        self.style_stat_file = f'{self.synthesizer}_style_stats.pkl'
        self.prosody_stat_file = f'{self.synthesizer}_prosody_stats.pkl'
        if style_dir and prosody_dir:
            self.style_data = self.gen_style_stats(style_dir)
            self.prosody_data = self.gen_prosody_stats_from_filenames(prosody_dir)
             
        else:
            try:
                self.style_data = pd.read_pickle(self.style_stat_file)
                self.prosody_data =  pd.read_pickle(self.prosody_stat_file)
              
            except:
                print("control stats not found; python embedding_control.py extract <prosody_dir> <style_dir> to generate.")
                sys.exit(0)

        self.feats = self.prosody_data.columns[3:]
        #self.feats = self.style_data.columns[4:]
        self.styles = self.style_data['style'].unique()
        #self.visualize_styles()
        self.gen_controls()


    def visualize_styles(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        # do a PCA on the style embeddings and plot scatterplot colored by style
        # limit the styles to 'neutral', 'calm', 'projected', 'whisper'
        df = self.style_data[self.style_data['style'].isin(['default', 'calm', 'projected', 'whisper'])]
        X = np.array(df['spkr_embedding'].tolist())
        pca = PCA()
        X_PCA = pca.fit_transform(X)
        
        df['spkr_embedding_PCA'] = list(X_PCA)
        print(df.head())
        sns.scatterplot(x=X_PCA[:,0], y=X_PCA[:,1], hue=df['style'])
        
        # mark mean of each style in the PCA space and plot, use same color as in previous plot for each style
        means = {}
        for style in self.styles:
            means[style] = np.array(df[df['style']==style]['spkr_embedding_PCA'].mean())
            print(style, means[style])
        for style in ['default', 'calm', 'projected', 'whisper']:
            
            plt.scatter(means[style][0], means[style][1], color="black", marker='x') #,label=style, s=100, marker='x')
      
        # plot arrow from means['default'] to means['whisper']
        plt.arrow(means['default'][0], means['default'][1], means['whisper'][0]-means['default'][0], means['whisper'][1]-means['default'][1], color='black', head_width=0.1, head_length=0.1)
        plt.show()

    def get_embeddings(self, utt_wav):
        embedding = self.model.get_embeddings(utt_wav)
        return embedding.cpu().numpy().flatten()

        
        
    def get_prosodic_feats(self, utt_wav):
        sound = parselmouth.Sound(utt_wav)
        f0min = 75
        f0max = 500
        unit = "semitones re 100 Hz"
        pitch = parselmouth.praat.call(sound, "To Pitch", 0.0, f0min, f0max)  # create a praat pitch object
        f0_mean = parselmouth.praat.call(pitch, "Get mean", 0, 0, unit)  # get mean pitch
        f0_std = parselmouth.praat.call(pitch, "Get standard deviation", 0, 0, "semitones")  # get standard deviation
        
        ltas = parselmouth.praat.call(sound, "To Ltas", 100)
        harmonicity = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
        tilt = parselmouth.praat.call(ltas, "Get slope", 0,1000,1000,4000, 'energy')  # get mean pitch

        return f0_mean, f0_std, hnr, tilt
        
    
    def gen_style_stats(self, directory):
        wavs = glob.glob(directory+"/*.wav")
        if len(wavs)==0:
            print("No wavs found in directory.", directory)
            return None 
        rows = []
        max_utts = min(len(wavs),200)
        print("extracting embeddings and prosodic features from style data...")
        for i in tqdm(range(0, max_utts)):
            
            embedding = self.get_embeddings(wavs[i])
            f0_mean, f0_std, hnr,tilt = self.get_prosodic_feats(wavs[i])
            style = ""
            try:
                style = os.path.basename(wavs[i]).split("_")[1]
                
            except:
                pass
            row = {
                'filename': os.path.basename(wavs[i]),
                'style':str(style),
                'embedding': embedding,
                'f0mean':f0_mean,
                'f0std':f0_std,
                'spec':tilt,
                'noise':hnr
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_pickle(self.style_stat_file)
        return df
    
    

    def gen_prosody_stats_from_filenames(self,directory):

        wavs = sorted(glob.glob(directory+"/*.wav"))
        rows = []
        max_utts = min(len(wavs),3000)

        print("extracting embeddings from prosody manipulation data...")
        for i in tqdm(range(0, max_utts)):
            embedding = self.get_embeddings(wavs[i]) 
            
            # path/f0mean_2.0.wav
            #mod, val = os.path.basename(wavs[i])[:-4].split("_")
            # path/f0mean_2.0_reverb_0.22.wav
            mod2 = None
            try:
                _, mod1, val1, mod2, val2 = os.path.basename(wavs[i])[:-4].split("_")
            except:
                mod1, val1 = os.path.basename(wavs[i])[:-4].split("_")
           
            
            row = {
                'filename': os.path.basename(wavs[i]),
                'embedding': embedding,
                'f0mean':np.nan,'f0std':np.nan,'spec':np.nan,'rate':np.nan #,'reverb':np.nan,'noise':np.nan, 'comp':np.nan
            }
            row[mod1] = val1
            if mod2:
                row[mod2] = val2
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_pickle(self.prosody_stat_file)
        return df
    
    def gen_controls(self):
        df = self.style_data
        neutral_style_df = self.style_data[self.style_data['style']=='default']
        embedding = 'spkr_embedding'
        #embedding = 'embedding'
        
        
        self.mean_style_embedding = np.array(neutral_style_df[embedding].mean())


        df = self.prosody_data
        #df = self.style_data
        
        
        #X = np.array(df['spkr_embedding'].tolist())
        X = np.array(df[embedding].tolist())
        print(df.head())
        self.embedding_mean = np.mean(X, axis=0)
        self.embedding_std = np.std(X, axis=0)
        X = (X-self.embedding_mean) #/self.spkr_embedding_std
        
        df['embedding_norm'] = X.tolist()
        
        
        for feat in self.feats:
       
           
            y = np.array(df.dropna(subset=[feat])[feat], dtype=float)
            X = np.array(df.dropna(subset=[feat])['embedding_norm'].tolist())
            
            #print(feat, np.min(y), np.max(y), np.std(y))
            
           
            self.control_coeffs[feat]=LinearRegression().fit(X, y).coef_
            self.control_weights[feat]=1./np.max(np.abs(self.control_coeffs[feat]))
            """
            # Calculate correlations and filter coefficients:
            correlations = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
            print(np.sort(abs(correlations))) #)[:-50])
            #input()
            self.control_coeffs[feat][np.abs(correlations) < 0.1] = 0
            #self.control_coeffs[feat][:] = 0
            
            self.feat_std[feat] = np.std(y)
            """
        

    def mod_style(self, key="style", val="neutral"):
        try:
            style_specific_df = self.style_data[self.style_data[key]==val]
        except:
            print("Style not found:", val)
            return 0
        print(style_specific_df.head())
        style_embedding = np.array(style_specific_df['spkr_embedding'].mean())
        #if not self.gpt_embedding:
        return style_embedding - self.mean_style_embedding #, None
        #else:
        #    gpt_embedding = np.array(style_specific_df['gpt_embedding'].mean())

        #    return style_embedding - self.mean_style_embedding, gpt_embedding-self.mean_style_embedding_gpt
       
    

    def mod_embeddings(self, feature, value, orthogonal=False, embedding=None):
        return self.mod_speaker_embedding(feature, value, orthogonal, embedding)


    def mod_speaker_embedding(self, feature, value, orthogonal=False, base_embedding = None):
       
        y = self.control_coeffs[feature]
        if orthogonal:
            coefficients_dropped = dict(self.spkr_control_coeffs)
            del(coefficients_dropped[feature])
            X = np.stack(coefficients_dropped.values()).T     
            Py = X @ np.linalg.inv(X.T @ X) @ X.T @ y
            Oy = y - Py
        else:
            Oy = y
        #embedding = base_embedding + self.spkr_embedding_std*float(value)*Oy*self.spkr_control_weights[feature]
        embedding = base_embedding + float(value)*Oy*self.control_weights[feature]
        #embedding = base_embedding + (float(value) / self.feat_std[feature]) * Oy 
           
        return embedding

    

    def speak(self, text, embedding, fname = "tmp.wav", play=False, second_embedding=None):
        embedding = torch.tensor(embedding, dtype=torch.float32)
        embedding = embedding.unsqueeze(0).to(self.device)
        out = self.model.inference(text, embedding, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1)    
        torchaudio.save(fname, torch.tensor(out).unsqueeze(0), 24000)
        if play:
            os.system("play "+fname)
        return(out)
                



if __name__ == "__main__":
    

    print("extract stats for contols: python embedding_control.py extract <prosody_dir> <style_dir> <tts_model_name>")
    print("to test: python embedding_control.py <wav_file> <tts_model_name>")
    tts_model =sys.argv[-1]
    if len(sys.argv) == 5 and sys.argv[1] == "extract":
        syn = StyleControls(tts_model, prosody_dir=sys.argv[2], style_dir=sys.argv[3])
        print("Control stats extracted.")
        sys.exit(0)
    else:
        syn = StyleControls(tts_model)

    embedding = syn.get_embeddings(sys.argv[1])


    text = "In theory, what I really want to do is figure out next-generation learning algorithms."

    
    for style in syn.styles:
        mod_embedding = embedding + syn.mod_style("style", style)
        wav = syn.speak(text, mod_embedding, play=True)
        
        
    for i in range(-5,5,1):
        print(i*0.5)
        embedding_mod = syn.mod_embeddings("f0mean", float(i), False, embedding)
        wav = syn.speak(text, embedding_mod, play=True)
            
    sys.exit(0)                       
