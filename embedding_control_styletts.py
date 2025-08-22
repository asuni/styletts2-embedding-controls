import os, sys, glob
import numpy as np

import torch
import torchaudio


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm


class StyleControls():
    def __init__(self, prosody_dir=None, style_dir=None, synthesizer="styletts2"):
        self.device = "cuda"
        self.synthesizer = synthesizer
        import styletts
        self.model = styletts.StyleTTS()

        self.feats = [] 
        self.styles = []

       
        self.control_coeffs = {}
        self.prosody_vectors = {}

        self.style_stat_file = f'{self.synthesizer}_style_stats.pkl'
        self.prosody_stat_file = f'{self.synthesizer}_prosody_stats.pkl'
        if style_dir and prosody_dir:
            self.style_data = self.gen_style_stats(style_dir)
            self.prosody_data = self.gen_prosody_stats_from_filenames(prosody_dir)
             
        else:
            try:
            
                self.style_data = pd.read_pickle(self.style_stat_file)
                self.prosody_data =  pd.read_pickle(self.prosody_stat_file)
                print("Unique styles:", self.style_data["style"].unique())
            except:
                print("control stats not found; python embedding_control.py extract <prosody_dir> <style_dir> to generate.")
                sys.exit(0)
        
        self.feats = self.prosody_data.columns[2:]
       
        # remove read styles
        self.style_data= self.style_data[~self.style_data['style'].str.endswith('-r', na=False)]

        # and or join conversational and read styles
        self.style_data['style'] = self.style_data['style'].str[:-2]
        self.styles = self.style_data['style'].unique()
        self.gen_controls()
    

    def get_svm_style_vector(self, 
                         target_style, neutral_style='default',
                         embedding_col = 'embedding', 
                         style_col = 'style',
                         svm_C: float = 1.0): 
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        df = self.style_data
        # 1. Filter data for the two styles
        neutral_df = df[df[style_col] == neutral_style]
        target_df = df[df[style_col] == target_style]

        if neutral_df.empty:
            raise ValueError(f"No data found for neutral style: '{neutral_style}'")
        if target_df.empty:
            raise ValueError(f"No data found for target style: '{target_style}'")

        #print(f"Found {len(neutral_df)} samples for neutral style '{neutral_style}'")
        #print(f"Found {len(target_df)} samples for target style '{target_style}'")

        #Prepare data for scikit-learn
        try:
            # Convert list-like embeddings to a NumPy array
            X_neutral = np.array(neutral_df[embedding_col].tolist())
           
            X_target = np.array(target_df[embedding_col].tolist())
            neutral_mean = np.mean(X_neutral, axis=0)
            target_mean = np.mean(X_target,axis=0)
            distance = np.linalg.norm(neutral_mean - target_mean)
          
            # Check embedding dimensions consistency implicitly via vstack
            X = np.vstack((X_neutral, X_target))
            
            # Create labels: 0 for neutral, 1 for target style
            y = np.concatenate((np.zeros(len(X_neutral)), np.ones(len(X_target)))) 
            
            if X.ndim != 2:
                raise ValueError("Embeddings could not be stacked into a 2D array.")
            embedding_dim = X.shape[1]
            print(f"Embeddings dimension detected: {embedding_dim}")

        except Exception as e:
            raise ValueError(f"Error processing embeddings: {e}. Ensure '{embedding_col}' contains consistent list/array data.")

        # Scale the data
        # SVMs are sensitive to feature scaling
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        X_scaled = X
       

        # Train Linear SVM
       
        svm_model = SVC(kernel='linear', C=svm_C, probability=False, random_state=42) 
        svm_model.fit(X_scaled, y)
        

        # 5. Extract hyperplane normal vector (coefficients)
        normal_vector = svm_model.coef_[0] 

        # 6. Normalize to get the unit vector
        norm = np.linalg.norm(normal_vector)
        if norm < 1e-9: # Check for near-zero norm
            warnings.warn("SVM coefficients have near-zero norm. The styles might be inseparable or identical.")
            # Return the zero vector or raise error depending on desired behavior
            # Returning zero vector here to indicate no direction found
            return np.zeros_like(normal_vector) 
            # raise ValueError("SVM coefficients have zero norm, cannot normalize. Styles might be inseparable.")

        unit_normal_vector = normal_vector / norm
        #print(f"Calculated unit normal vector (style vector) with shape: {unit_normal_vector.shape}")
        
        return unit_normal_vector, distance

    # Function to calculate the geometric median using Weiszfeld's algorithm
    def _geometric_median(self, X, eps=1e-6):
        
        from scipy.spatial.distance import euclidean
        # Initialize the median as the mean of the points
        y = np.mean(X, axis=0)
    
        while True:
            # Compute the distances from the current median to all points
            distances = np.linalg.norm(X - y, axis=1)
            # Avoid division by zero by adding a small value to zero distances
            non_zero = distances > eps
            distances[~non_zero] = eps
            # Calculate the new median
            y_new = np.sum(X[non_zero] / distances[non_zero, None], axis=0) / np.sum(1.0 / distances[non_zero])
            # Check for convergence
            if euclidean(y, y_new) < eps:
                break
            y = y_new

        return y


    def get_embeddings(self, utt_wav):
        embedding = self.model.get_embeddings(utt_wav)
        return embedding.cpu().numpy().flatten()

    def gen_style_stats(self, directory):
        
        wavs = glob.glob(directory+"/*.wav")
        
        if len(wavs)==0:
            print("No wavs found in directory.", directory)
            return None
        
        rows = []
        print("extracting embeddings and prosodic features from style data...")
        for i in tqdm(range(0, len(wavs))):
            try:
                embedding = self.get_embeddings(wavs[i])
            except:
                continue
          
            style = ""
            try:
                # adapt to your filenaming scheme
                style = os.path.basename(wavs[i]).split("_")[1]
               
            except:
                print("Error extracting style from filename:", os.path.basename(wavs[i]))

            row = {
                'filename': os.path.basename(wavs[i]),
                'style':str(style),
                'embedding': embedding,   
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_pickle(self.style_stat_file)
        return df
    
    

    def gen_prosody_stats_from_filenames(self,directory):

        wavs = sorted(glob.glob(directory+"/*.wav"))
        rows = []

        print("extracting embeddings from prosody manipulation data...")
        for i in tqdm(range(0, len(wavs))):
            embedding = self.get_embeddings(wavs[i]) 
            
            mod1, val1 = os.path.basename(wavs[i])[:-4].split("_")
           
            row = {
                'filename': os.path.basename(wavs[i]),
                'embedding': embedding,
                'f0mean':np.nan,'f0std':np.nan,'spec':np.nan,'rate':np.nan
            }
            row[mod1] = val1
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_pickle(self.prosody_stat_file)
        return df
    
    def gen_controls(self):
    
        df = self.style_data
        neutral_style_df = self.style_data[self.style_data['style']=='default']
        
        self.mean_style_embedding = self._geometric_median(np.array(neutral_style_df['embedding'].tolist()))
    

        df = self.prosody_data


        X = np.array(df['embedding'].tolist())
        self.embedding_mean = np.mean(X, axis=0)
        self.embedding_std = np.std(X, axis=0)
        X = (X-self.embedding_mean)/self.embedding_std
        
        df['embedding_norm'] = X.tolist()
        for feat in self.feats:
       

            y = np.array(df.dropna(subset=[feat])[feat], dtype=float)
            X = np.array(df.dropna(subset=[feat])['embedding'].tolist())
            #X = np.array(df.dropna(subset=[feat])['embedding_norm'].tolist())
            print(feat, y.shape, X.shape)
            #print(feat, np.min(y), np.max(y), np.std(y))
            # centering
            if feat in ["f0mean", "f0std", "rate"]:
                y = y-1
            
                
            model = LinearRegression().fit(X, y)
            # Store coefficients and intercept
            self.control_coeffs[feat] = model.coef_
        
            norm_sq = np.dot(self.control_coeffs[feat], self.control_coeffs[feat])
            self.prosody_vectors[feat] =  self.control_coeffs[feat] / norm_sq
        """
        # estimate prosody control coefficients
        for feat in self.feats:
            y = np.array(df.dropna(subset=[feat])[feat], dtype=float)
            X = np.array(df.dropna(subset=[feat])['embedding_norm'].tolist())
            self.control_coeffs[feat]=LinearRegression().fit(X, y).coef_
            self.control_weights[feat]=1./np.max(np.abs(self.control_coeffs[feat]))
         """  
        

    def get_style_vector(self, key="style", val="neutral", use_median=True):
        try:
            style_specific_df = self.style_data[self.style_data[key]==val]
        except:
            print("Style not found:", val)
            return 0

        if use_median:
            style_embedding = self._geometric_median(np.array(style_specific_df['embedding'].tolist()))
        else:
            style_embedding = np.array(style_specific_df['embedding'].mean())

        return style_embedding - self.mean_style_embedding 


   
    def add_prosody_vector(self, feature, value, orthogonal=False, base_embedding = None):
       
        return base_embedding + float(value)*self.prosody_vectors[feature]
        
           
      


    def speak(self, text, embedding, fname = "tmp.wav", play=False, second_embedding=None, alpha=0.1, beta=0.9, emb_scale=1.):
        embedding = torch.tensor(embedding, dtype=torch.float32)
        embedding = embedding.unsqueeze(0).to(self.device)
        out = self.model.inference(text, embedding, alpha=alpha, beta=beta, diffusion_steps=20, embedding_scale=emb_scale)    
        torchaudio.save(fname, torch.tensor(out).unsqueeze(0), 24000)
        if play:
            os.system("play -q "+fname)
        return(out)
                



if __name__ == "__main__":
    

    print("extract stats for contols: python embedding_control.py extract <prosody_dir> <style_dir>")
    print("to test: python embedding_control.py <wav_file>")
    
    if len(sys.argv) == 4 and sys.argv[1] == "extract":
        syn = StyleControls(prosody_dir=sys.argv[1], style_dir=sys.argv[2])
        print("Control stats extracted.")
        sys.exit(0)
    else:
        syn = StyleControls()
   
    embedding = syn.get_embeddings(sys.argv[1])

    

   
    text = "A quick brown fox jumped over the lazy old dog."
    for style in syn.styles:
        if style == "default":
            continue    
        print(style)
        mod_embedding = embedding + syn.get_style_vector("style", style) #*-1.5
        wav = syn.speak(text, mod_embedding, play=True)
        #style_vec, alpha = syn.get_svm_style_vector(style)
        #mod_embedding= embedding + style_vec*alpha #*-1.5
        #wav = syn.speak(text, mod_embedding, play=True)
     
    for i in range(-5,5,1):
        print(i*0.5)
        embedding_mod = syn.add_prosody_vector("f0mean", float(i), False, embedding)
        wav = syn.speak(text, embedding_mod, play=True)
            
    sys.exit(0)                       
