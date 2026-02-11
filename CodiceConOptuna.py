"""
EVALITA MultiPRIDE 2026 - Challenge Solution: Hate Speech Detection
Team: GRUPPETTOZZO
Autori: Federico Traina, Alessandro Santoro, Gabriele Greco

DESCRIZIONE DEL MODULO:
Questo script implementa una pipeline di AutoML (Machine Learning Automatizzato) end-to-end.
Utilizza il framework Optuna per ottimizzare gli iperparametri di modelli Transformer
(BERT, RoBERTa, XLM-R) in uno scenario multilingua (IT, EN, ES).
Il codice gestisce:
1. Caricamento dinamico dei dati.
2. Preprocessing sensibile al contesto (gestione Emoji).
3. Training con precisione mista (FP16) per efficienza di memoria.
4. Strategie di Pooling avanzate (Concatenazione di feature).
"""

# ==================================================================================
# IMPORTAZIONE LIBRERIE E GESTIONE DIPENDENZE
# ==================================================================================

import os   # Interazione con il sistema operativo (file system, variabili d'ambiente)
import sys  # Accesso ai parametri di sistema e moduli
import gc   # Garbage Collector: CRUCIALE in PyTorch. Ci permette di liberare
            # manualmente la memoria RAM e VRAM (GPU) tra un trial di Optuna e l'altro,
            # prevenendo errori di tipo "CUDA Out of Memory".

import random
import time
import copy # Utilizzato per effettuare la 'deep copy' dei pesi del modello migliore
            # durante l'Early Stopping, evitando di salvare solo il riferimento in memoria.
import json
import numpy as np
import pandas as pd # Manipolazione tabellare dei dataset (CSV)

# Framework di Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import optuna 
import emoji  # Libreria specifica per la manipolazione semantica delle emoji (Demojization)
import nltk   # Natural Language Toolkit per stopword e tokenizzazione classica
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight # Fondamentale per dataset sbilanciati
from sklearn.metrics import f1_score # Metrica ufficiale della challenge Evalita

from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    get_linear_schedule_with_warmup, # Scheduler per variare il Learning Rate dinamicamente
    logging as hf_logging
)
try:
    from pytorch_optimizer import RAdam, Lion
except ImportError:
    pass 

# Configurazione Logging:
# Impostiamo la verbosità di HuggingFace su 'ERROR' per evitare che la console 
# venga inondata di warning non critici (es. pesi non inizializzati nel fine-tuning).
hf_logging.set_verbosity_error()

# Rilevamento automatico dell'ambiente di esecuzione.
# Se stiamo girando su Google Colab o Kaggle , installiamo silenziosamente (-q)
if 'google.colab' in sys.modules or os.path.exists('/kaggle'):
    print("☁️ Ambiente Cloud rilevato: Installazione dipendenze mancanti...")
    os.system('pip install -q emoji optuna pytorch-optimizer')

# Definizione dinamica del PATH dei dati.

if os.path.exists('/kaggle/input/dataaumentato'):
    BASE_PATH = Path('/kaggle/input/dataaumentato')
elif os.path.exists('/kaggle/input/dataset-multipride'):
    BASE_PATH = Path('/kaggle/input/dataset-multipride')
else:
    BASE_PATH = Path('./dataset')

# Definizione della directory di output.
# Su Kaggle/Colab, solo la cartella 'working' è scrivibile.
OUTPUT_DIR = Path('/kaggle/working/results_ALL_LANGS_FINAL')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Percorso Dati Input: {BASE_PATH}")
print(f"Percorso Output: {OUTPUT_DIR}")

# ==================================================================================
# 3. IPERPARAMETRI GLOBALI E RISORSE
# ==================================================================================

SEED = 42           # Seme per la riproducibilità (vedi funzione set_seed)
MAX_LENGTH = 128    # Lunghezza massima della sequenza di token (troncamento/padding)
                    # 128 è un compromesso standard tra performance e memoria per i tweet.
NUM_WORKERS = 2     # Numero di sottoprocessi per il caricamento dati parallelo (DataLoader)
N_TRIALS = 20        # Numero di esperimenti (trial) che Optuna eseguirà per ogni configurazione
N_EPOCHS = 10        # Numero massimo di epoche di addestramento per trial
PATIENCE = 3        # Soglia per l'Early Stopping: se il modello non migliora per 3 epoche, stop.


# ==================================================================================
#  CONFIGURAZIONE LINGUE E SELEZIONE MODELLI 
# ==================================================================================

LANG_CONFIG = {
    # 🇮🇹 ITALIANO
    # Modello: UmBERTo (addestrato su CommonCrawl ITA).
    # Variante scelta: 'lupobricco/...', una versione già fine-tunata su hate speech.
    'IT': {
        'model': "lupobricco/umBERTo_fine-tuned_hate_offensivity",
        'nltk_lang': 'italian',
        'file_prefix': 'train_it',
        'is_multi': False
    },
    # 🇬🇧 INGLESE
    # Modello: BERT Base Uncased.
    # Motivazione: Lo standard de-facto per l'inglese. 'Uncased' perché l'hate speech
    # spesso usa MAIUSCOLE a caso, quindi normalizzare aiuta.
    'EN': {
        'model': "bert-base-uncased", 
        'nltk_lang': 'english',
        'file_prefix': 'train_en',
        'is_multi': False
    },
    # 🇪🇸 SPAGNOLO
    'ES': {
        'model': "dccuchile/bert-base-spanish-wwm-cased", 
        'nltk_lang': 'spanish',
        'file_prefix': 'train_es',
        'is_multi': False
    },
    # MULTILINGUA
    # Modello: XLM-RoBERTa (Cross-lingual Language Model).
    'MULTI': {
        'model': "xlm-roberta-base", 
        'nltk_lang': 'multi', 
        'file_prefix': 'train_multi', # Placeholder, userà MULTI_DATASETS_PATHS
        'is_multi': True
    }
}


# Controllo e download risorse NLTK.
# Scarichiamo stopwords e tokenizzatori solo se non sono già presenti nel sistema
# per evitare overhead di rete ad ogni avvio.
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# ==================================================================================
#  4. RIPRODUCIBILITÀ 
# ==================================================================================

def set_seed(seed: int):

    random.seed(seed)          # Per le funzioni random standard di Python
    np.random.seed(seed)       # Per le operazioni matriciali di NumPy
    torch.manual_seed(seed)    # Per i tensori PyTorch su CPU
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # Per i tensori su tutte le GPU disponibili
        
        # Configurazioni avanzate per il backend CUDA (NVIDIA):
        # 'deterministic = True' forza l'uso di algoritmi deterministici (riproducibili) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Applicazione del seed
set_seed(SEED)

# Selezione automatica del dispositivo di calcolo
# Usa la GPU (CUDA) se disponibile per accelerare il training, altrimenti CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Dispositivo di calcolo configurato: {DEVICE}")

# ==================================================================================
# 5. GENERAZIONE DATASET MULTILINGUA (DATA FUSION)
# ==================================================================================

def create_multilingual_datasets() -> Dict[str, Path]:
    """
    Genera i dataset per il task 'MULTI' (Cross-lingual) concatenando i corpus
    di addestramento di Italiano (IT), Inglese (EN) e Spagnolo (ES).
    
    Gestisce tre scenari di dati (Data Augmentation):
    1. Originale: Nessuna modifica.
    2. Aug_Full_A: Dataset raddoppiato con Back-Translation su TUTTI i campioni.
    3. Aug_Bal_B: Dataset bilanciato aumentando SOLO la classe minoritaria.
    """
    print("\n [DATA PREP] Creazione Dataset Multilingua (Merge IT+EN+ES)...")
    
    # Mappa dei suffissi per identificare automaticamente i file generati 
    # dallo script di Data Augmentation.
    files_map = {
        '1_Original':   ('',                'train_multi.csv'),
        '2_Aug_Full_A': ('_aug_A_full',     'train_multi_aug_A_full.csv'),
        '3_Aug_Bal_B':  ('_aug_B_balanced', 'train_multi_aug_B_balanced.csv')
    }
    
    langs = ['it', 'en', 'es']
    generated_paths = {}

    for key, (suffix, out_name) in files_map.items():
        dfs = []
        missing = False
        
        # Iteriamo su ogni lingua per cercare il file corrispondente alla variante corrente
        for lang in langs:
            fname = f"train_{lang}{suffix}.csv"
            fpath = BASE_PATH / fname
            
            if fpath.exists():
                d = pd.read_csv(fpath)
                d['lang_source'] = lang 
                dfs.append(d)
            else:
                print(f"File mancante per variante {key}: {fname}")
                missing = True
        
        # Se abbiamo trovato i file per tutte le lingue, procediamo al merge
        if not missing and dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            out_path = OUTPUT_DIR / out_name
            merged_df.to_csv(out_path, index=False)
            generated_paths[key] = out_path
            print(f"Creato dataset unificato: {out_name} ({len(merged_df)} righe)")
        else:
            print(f"Impossibile creare {out_name} (dataset incompleti)")
            
    return generated_paths

MULTI_DATASETS_PATHS = create_multilingual_datasets()


# ==================================================================================
# PREPROCESSING VARIANTS
# ==================================================================================

# Definiamo le strategie di pulizia per verificare l'ipotesi che le emoji contengano
# segnali di odio o rivendicazione (reclamation).

PREPROCESSING_VARIANTS = {
    # V1: Controllo. Minima interferenza, lasciamo il testo quasi grezzo.
    'V1_Minimal': {
        'remove_emojis': False, 
        'convert_emojis': False, 
        'remove_stopwords': False
    },
    # V2: Trasformiamo le emoji in testo (es. 🏳️‍🌈 -> :bandiera_arcobaleno:).
    # Ipotesi: Questo aiuta BERT a "leggere" l'emoji come una parola semanticamente ricca.
    'V2_Standard': {
        'remove_emojis': False, 
        'convert_emojis': True,  
        'remove_stopwords': False
    },
    # V3: Approccio Classico. Rimuoviamo tutto ciò che non è testo standard.
    'V3_Aggressive': {
        'remove_emojis': True,  
        'convert_emojis': False, 
        'remove_stopwords': True
    }
}

# Variabili globali per mantenere lo stato della configurazione corrente durante i loop
CURRENT_CLEANING_PARAMS = {}
CURRENT_LANG = 'IT'

# ==================================================================================
# REGOLARIZZAZIONE: EARLY STOPPING
# ==================================================================================

class EarlyStopping:

    def __init__(self, patience=3, min_delta=0, verbose=False):
        self.patience = patience      # Quante epoche aspettare prima di fermarsi
        self.min_delta = min_delta    # Miglioramento minimo richiesto per resettare il counter
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None 

    def __call__(self, score, model):
        # Primo step: inizializzazione
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
        
        # Se il punteggio non migliora significativamente
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        # Se c'è un miglioramento
        else:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0

    def load_best_weights(self, model):
        """Ripristina i pesi della migliore epoca osservata nel modello attuale."""
        if self.best_model_state:
            model.load_state_dict(self.best_model_state)

# ==================================================================================
# TEXT CLEANER & DEMOJIZATION ENGINE
# ==================================================================================

class TextCleaner:
    """
    Implementa una logica di 'Demojization' localizzata. Invece di rimuovere le emoji
    o lasciarle come codici Unicode (che BERT spesso ignora), le traduciamo
    nella lingua target (es. 💔 -> 'cuore spezzato' in IT, 'broken heart' in EN).
    """
    # Database Stopwords caricato una sola volta per efficienza
    STOPWORDS_DB = {
        'italian': set(stopwords.words('italian')),
        'english': set(stopwords.words('english')),
        'spanish': set(stopwords.words('spanish')),
    }
    STOPWORDS_DB['multi'] = STOPWORDS_DB['italian'] | STOPWORDS_DB['english'] | STOPWORDS_DB['spanish']

    @staticmethod
    def clean(text):
        """
        Applica la pipeline di pulizia in base a CURRENT_CLEANING_PARAMS.
        """
        text = str(text)
        params = CURRENT_CLEANING_PARAMS 
        lang_key = LANG_CONFIG[CURRENT_LANG]['nltk_lang']
        
        # 1. Normalizzazione case (tutto minuscolo) per ridurre la sparsità del vocabolario, questa la facciamo sempre.
        text = text.lower()
        
        # 2. Gestione Emoji:
        if params.get('remove_emojis'):
            # Strategia V3: Rimozione totale (pulizia rumore)
            text = emoji.replace_emoji(text, replace='')
            
        elif params.get('convert_emojis'):
            # Strategia V2
            try:
                # Selezioniamo la lingua di destinazione per la traduzione dell'emoji
                if CURRENT_LANG == 'IT': code = 'it'
                elif CURRENT_LANG == 'ES': code = 'es'
                else: code = 'en' # Inglese usato per EN e come lingua franca per MULTI
                
                text = emoji.demojize(text, delimiters=(" ", " "), language=code)
            except:
                # Fallback sicuro all'inglese se la libreria non supporta la lingua
                text = emoji.demojize(text, delimiters=(" ", " "))
            
        # 3. Rimozione Stopwords 
        if params.get('remove_stopwords'):
            words = word_tokenize(text)
            sw_set = TextCleaner.STOPWORDS_DB.get(lang_key, set())
            text = " ".join([w for w in words if w.lower() not in sw_set])
            
        return text
    
    # ==================================================================================
# 10. ARCHITETTURA MODELLO:  BERT CLASSIFIER
# ==================================================================================

class AdvancedBERTClassifier(nn.Module):
    """
    Wrapper personalizzato attorno ai modelli Transformer (BERT, RoBERTa, XLM-R).
        
    1. Layer Freezing: Congelamento selettivo dei layer inferiori per preservare 
       la conoscenza linguistica di base (sintassi) e adattare solo i layer alti (semantica).
    2. Hybrid Pooling: Concatenazione di diverse rappresentazioni della frase.
    3. MLP Head: Un piccolo network neurale sopra il Transformer per la decisione finale.
    """
    def __init__(self, model_name, dropout_prob=0.2, freeze_layers=0, pooling_type='cls', mlp_hidden_size=256):
        super().__init__()
        self.pooling_type = pooling_type
        
        # Caricamento del backbone pre-addestrato (es. UmBERTo, XLM-R)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Rileviamo se è un modello XLM/RoBERTa perché non usano 'token_type_ids'
        self.is_xlm = 'xlm' in model_name.lower() or 'roberta' in model_name.lower()
        
        # ---------------------------------------------------------
        # A. STRATEGIA DI FREEZING
        # ---------------------------------------------------------
        # Se freeze_layers > 0, blocchiamo l'aggiornamento dei pesi nei primi N strati.
        if freeze_layers > 0:
            encoder = getattr(self.bert, 'encoder', None)
            if encoder and hasattr(encoder, 'layer'):
                for i, layer in enumerate(encoder.layer):
                    if i < freeze_layers:
                        for param in layer.parameters():
                            param.requires_grad = False
                            
        # ---------------------------------------------------------
        # B. CALCOLO DIMENSIONE INPUT CLASSIFICATORE
        # ---------------------------------------------------------
        hidden_size = self.bert.config.hidden_size # Di solito 768 per BERT-base
        
        # Se usiamo 'concat', l'input triplica (CLS + Mean + Max = 768 * 3 = 2304)
        input_dim = hidden_size * 3 if pooling_type == 'concat' else hidden_size
            
        # ---------------------------------------------------------
        # C. MLP CLASSIFICATION HEAD
        # ---------------------------------------------------------
        # Sostituiamo il layer lineare semplice con un MLP a due strati + Dropout
        # per aggiungere non-linearità alla decisione finale.
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),                 # Regolarizzazione
            nn.Linear(input_dim, mlp_hidden_size),    # Proiezione intermedia
            nn.ReLU(),                               
            nn.Dropout(dropout_prob),                 
            nn.Linear(mlp_hidden_size, 1)             # Output Logit (Binario)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """Passaggio in avanti (Forward Pass) della rete."""
    
        # Gestione input differenziata per BERT vs RoBERTa/XLM
        if self.is_xlm or token_type_ids is None:
             outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
             outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        # Estraiamo gli stati nascosti dell'ultimo layer (Batch, Seq_Len, Hidden_Dim)
        last_hidden = outputs.last_hidden_state
        
        # ---------------------------------------------------------
        # D. IMPLEMENTAZIONE POOLING STRATEGIES
        # ---------------------------------------------------------
        
        if self.pooling_type == 'cls':
            # STANDARD: Usa solo il vettore del primo token speciale [CLS]
            pooled = last_hidden[:, 0, :]
            
        elif self.pooling_type == 'mean':
            # AVERAGE POOLING: Media di tutti i token della frase.
            # Dobbiamo usare l'attention_mask per non includere i token di padding (zeri) nella media.
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            #Ottieni un volume di [16, 128, 768] dove ogni parola vera è un blocco di soli 1 e ogni parola di padding è un blocco di soli 0.
            #moltiplico last_hidden che ha dimensione 768 per input_mask con la maschera espansa
            
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
            #input_mask_expanded.sum(1): Calcola quanti token non-padding ci sono per ogni riga. 
            #Se una frase ha 10 parole reali, questo valore sarà 10.
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9) # Evita divisione per zero
            pooled = sum_embeddings / sum_mask
            
        elif self.pooling_type == 'concat':
            # HYBRID POOLING (La nostra proposta):
            # Concatena:
            # 1. CLS (Rappresentazione globale sintetica)
            # 2. Mean Pooling (Contesto medio)
            # 3. Max Pooling 
            
            cls_out = last_hidden[:, 0, :]
            
            # Calcolo Mean
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_out = sum_embeddings / sum_mask
            
            # Calcolo Max (mascherando il padding con un numero molto piccolo)
            last_hidden_masked = last_hidden.clone()
            last_hidden_masked[input_mask_expanded == 0] = -1e9
            max_out = torch.max(last_hidden_masked, 1)[0]
            
            # Concatenazione finale
            pooled = torch.cat((cls_out, mean_out, max_out), dim=1)
            
        else:
            # Fallback a CLS
            pooled = last_hidden[:, 0, :] 

        # Passaggio finale attraverso l'MLP
        return self.classifier(pooled)

# ==================================================================================
# LOSS FUNCTION : FOCAL LOSS
# ==================================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma          # Parametro di focalizzazione (ottimizzato da Optuna)
        self.pos_weight = pos_weight # Peso per bilanciare le classi (calcolato dal dataset)

    def forward(self, inputs, targets):
        # Calcolo Binary Cross Entropy di base (senza riduzione per applicare i pesi)
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calcolo probabilità (pt)
        probs = torch.sigmoid(inputs)
        targets = targets.type_as(probs)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        
        # Fattore Alpha (bilanciamento frequenza classi)
        if self.pos_weight is not None:
            alpha_factor = targets * self.pos_weight + (1 - targets)
        else:
            alpha_factor = targets * self.alpha + (1 - targets) * (1.0 - self.alpha)

        loss = alpha_factor * ((1.0 - p_t) ** self.gamma) * bce

        #media sul batch. la loss è un tensore di loss, una per ogni
        return loss.mean()

# ==================================================================================
# 12. DATA LOADING E GESTIONE CONTESTUALE (TASK A/B)
# ==================================================================================

def get_dataset_splits(csv_path, task_type):
    """
    Prepara i dati per l'addestramento.
    
    TASK B (Context-Aware):
    Se il task è 'B', questa funzione concatena il testo del tweet con la biografia
    dell'utente usando il token speciale [SEP].
    Input Modello: "testo tweet [SEP] CONTESTO: bio utente"
    
    Gestisce anche:
    - Pulizia del testo (TextCleaner)
    - Calcolo dei pesi per la Loss (per sbilanciamento)
    """
    if not csv_path.exists(): 
        print(f"⚠️ Errore: File dataset non trovato in {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    
    # Applicazione Preprocessing (Demojization, Lowercasing, ecc.)
    df['text'] = df['text'].apply(TextCleaner.clean)
    
    # --- LOGICA CONTESTUALE (TASK B) ---
    if task_type == 'B':
        # Gestione valori mancanti nella bio
        if 'bio' not in df.columns:
            df['bio'] = "" 
        else:
            df['bio'] = df['bio'].fillna('').astype(str).apply(TextCleaner.clean) 
        
        # Concatenazione con token separatore 
        df['combined_text'] = df['text'] + " [SEP] CONTESTO: " + df['bio']
    else:
        # Task A: Solo testo
        df['combined_text'] = df['text']

    # Rimozione duplicati per evitare Data Leakage (stesso testo in train e test)
    df.drop_duplicates(subset=['combined_text'], keep='first', inplace=True)
    
    # Rimozione righe vuote o corrotte
    df = df[df['combined_text'].str.strip().astype(bool)]
    
    X_text = df['combined_text']
    y = df['label'].astype(int)
    
    # --- STRATIFIED SPLIT ---
    # Garantisce che la proporzione di Hate/Non-Hate sia identica in train e test
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y, test_size=0.20, stratify=y, random_state=SEED
        )
    except ValueError:
        # Fallback se le classi sono troppo piccole per stratificare
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y, test_size=0.20, random_state=SEED
        )

    # Calcolo pesi delle classi per la Weighted Loss / Focal Loss
    classes = np.unique(y_train)

    # n_campioni / n_classi*n_campioni_i 
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    
    return X_train, X_test, y_train, y_test, torch.tensor(cw, dtype=torch.float)

def create_torch_datasets(X_tr, X_te, y_tr, y_te, model_name):
    """
    Tokenizzazione e conversione in Dataset PyTorch ottimizzati.
    """
    # Carichiamo il tokenizzatore specifico del modello (es. UmBERTo tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tok(batch):
        # Padding='max_length' e Truncation=True garantiscono tensori di dimensione fissa
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH)
    
    # Creazione Dataset HuggingFace
    ds_tr = Dataset.from_pandas(pd.DataFrame({'text': X_tr, 'label': y_tr}))
    ds_te = Dataset.from_pandas(pd.DataFrame({'text': X_te, 'label': y_te}))
    
    # Applicazione tokenizzazione in batch
    #ogni riga del datset ora contiene le colonne di sotto
    ds_tr = ds_tr.map(tok, batched=True).remove_columns(['text'])
    ds_te = ds_te.map(tok, batched=True).remove_columns(['text'])
    
    # Formattazione per PyTorch (Tensori)
    # token_type_ids serve solo per alcuni modelli (BERT), non per RoBERTa
    cols = ['input_ids', 'attention_mask', 'label', 'token_type_ids'] 

    #token_type_ids (Opzionale): Presente solo per modelli come BERT. Indica al modello a quale "segmento" appartiene il testo (molto utile nel Task B dove hai Testo + Bio).
    
    cols_tr = [c for c in cols if c in ds_tr.column_names]
    cols_te = [c for c in cols if c in ds_te.column_names]
    
    ds_tr.set_format("torch", columns=cols_tr)
    ds_te.set_format("torch", columns=cols_te)
    
    return ds_tr, ds_te


# ==================================================================================
# 13. TRAINING
# ==================================================================================

def train_one_epoch(model, loader, optimizer, loss_fn, scaler, scheduler):
    """
    Esegue un'epoca di addestramento completa.
    
    DETTAGLI TECNICI:
    - Mixed Precision (FP16): Utilizza `autocast` e `GradScaler` per eseguire le operazioni
      in 16-bit dove possibile. Questo riduce l'uso della VRAM del 40-50% e velocizza
      il training su GPU moderne (Tensor Cores), permettendo batch size più grandi.
    - Gradient Clipping: Taglia i gradienti con norma > 1.0 per prevenire
      il problema dell'Exploding Gradient, comune nei Transformer profondi.
    """
    model.train() 
    total_loss = 0
    
    for batch in loader:
        #pulisci i gradiente dello step predecente
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE).float().unsqueeze(1) # Unsqueeze per matching dimensioni (Batch, 1)
        #Questo serve perché l'output del modello (i logits) ha quella forma, e per calcolare la Loss le dimensioni devono combaciare perfettamente.
        
        # Gestione opzionale token_type_ids (solo per BERT, non per RoBERTa/XLM)
        token_type_ids = batch.get('token_type_ids')
        if token_type_ids is not None: token_type_ids = token_type_ids.to(DEVICE)

        #FORWARD PASS (Precisione Mista)
        with torch.amp.autocast():
            logits = model(input_ids, mask, token_type_ids)
            #I logits sono i valori grezzi (prima della Sigmoid). Non applichiamo la Sigmoid nel modello perché le funzioni di Loss moderne sono più stabili numericamente se ricevono i valori grezzi.
            loss = loss_fn(logits, labels)
        
        # --- BACKWARD PASS (Scaled) ---
        # Scaliamo la loss per evitare underflow numerici del formato FP16
        #: In FP16, se un gradiente è ad esempio 0.00001, verrebbe arrotondato a 0 (Underflow).
        # Lo scaler lo moltiplica per un fattore enorme (es. $2^{16}$). Questo "gonfia" i gradienti rendendoli rappresentabili. .backward() calcola poi la derivata della Loss rispetto a ogni singolo parametro del modello.
        scaler.scale(loss).backward()

        # Torniamo a 32 bit i gradienti
        scaler.unscale_(optimizer) 

        #Se il backward() serve a capire in che direzione muoversi, 
        #il clip_grad_norm_ serve a decidere quanto grande deve essere il passo da fare, 
        #evitando che il modello faccia "salti nel vuoto".
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        
        # Aggiornamento pesi
        scaler.step(optimizer)
        scaler.update()
        
        # Aggiornamento Learning Rate (Scheduler)
        if scheduler is not None:
            scheduler.step()
            
        total_loss += loss.item()
        
    return total_loss / len(loader)

def evaluate(model, loader):
    """
    Valutazione del modello sul Validation Set.
    Calcola l'F1-Score Macro.
    """
    model.eval() # Disabilita Dropout per inferenza deterministica
    preds, labels = [], []
    
    with torch.no_grad(): 
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            token_type_ids = batch.get('token_type_ids')
            if token_type_ids is not None: token_type_ids = token_type_ids.to(DEVICE)
            
            with torch.amp.autocast():
                logits = model(input_ids, mask, token_type_ids)
            
            # Soglia di classificazione a 0.5 (Standard per classificazione binaria)
            preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int))
            labels.extend(batch['label'].numpy())
            
    # Zero_division=0 evita warning se il modello non predice mai una classe
    #La Macro F1 dà lo stesso peso a ogni classe, indipendentemente da quanti esempi ci sono. 
    #Se il modello ignora la classe "Odio", il suo punteggio Macro crollerà drasticamente, anche se azzecca tutti i "Neutri".
    #f1 macro= f1odio+f1nonodio/2
    return f1_score(labels, preds, average='macro', zero_division=0)

# ==================================================================================
# 14. OPTUNA (Hyperparameter Tuning)
# ==================================================================================

def objective(trial, train_ds, test_ds, weights, model_name):
    """
    Funzione obiettivo che Optuna cercherà di massimizzare.
    Definisce lo spazio di ricerca (Search Space) degli iperparametri.
    """
    # ---------------------------------------------------------
    # A. DEFINIZIONE SPAZIO DI RICERCA
    # ---------------------------------------------------------
    
    # Learning Rate: campionamento logaritmico. È più probabile trovare buoni LR
    # nell'ordine di 1e-5 che 5e-5.
    lr = trial.suggest_float('lr', 1e-5, 5e-5, log=True)
    
    # Batch Size: 16 o 32 (limitato dalla VRAM)
    batch_size = trial.suggest_categorical('bs', [16, 32])
    
    # Dropout: Regolarizzazione per evitare overfitting
    dropout = trial.suggest_float('dr', 0.1, 0.5) 
    
    # Architettura Dinamica: Quanti layer congelare? Che pooling usare?
    freeze_layers = trial.suggest_int('freeze_layers', 0, 6)
    pooling_type = trial.suggest_categorical('pooling', ['cls', 'mean', 'concat']) 
    mlp_hidden = trial.suggest_categorical('mlp_hidden', [128, 256]) 
    
    # Loss Function e Optimizer
    loss_name = trial.suggest_categorical('loss', ['BCE', 'Focal'])
    
    # Se la libreria 'Lion' è disponibile, Optuna può sceglierla, altrimenti fallback su AdamW
    optim_name = trial.suggest_categorical('optimizer', ['AdamW', 'Lion']) if 'Lion' in globals() else 'AdamW'
    # ---------------------------------------------------------
    # B. SETUP DEL TRIAL
    # ---------------------------------------------------------
    # Istanziazione del modello con i parametri suggeriti dal trial corrente
    model = AdvancedBERTClassifier(model_name, dropout, freeze_layers, pooling_type, mlp_hidden).to(DEVICE)
    
    # Setup Ottimizzatore
    if optim_name == 'Lion' and 'Lion' in globals():
        optimizer = Lion(model.parameters(), lr=lr, weight_decay=1e-2)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=True)
    
    # Learning Rate Scheduler con Warmup 
    total_steps = len(train_loader) * N_EPOCHS
    warmup_steps = int(total_steps * 0.1) 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    scaler = GradScaler() # Per Mixed Precision

    pos_weight = (weights[1] / weights[0]).to(DEVICE) # Peso per bilanciamento classi
    


    #Se Optuna deve decidere quale "motore" (Loss) funziona meglio, 
    # entrambi i motori devono partire con la stessa benzina (il bilanciamento dei dati). 
    # Se passassi il peso solo a una, la sfida tra le due sarebbe truccata 
    # e non sapresti mai se una vince perché è "più intelligente" o solo perché è "più bilanciata".

    # Setup Loss Function
    if loss_name == 'Focal':
        # Anche il gamma della Focal Loss viene ottimizzato
        focal_gamma = trial.suggest_float('gamma', 0.5, 3.0)
        loss_fn = FocalLoss(pos_weight=pos_weight, gamma=focal_gamma)
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # ---------------------------------------------------------
    # C. TRAINING LOOP CON PRUNING
    # ---------------------------------------------------------
    early_stopper = EarlyStopping(patience=PATIENCE)
    best_f1 = 0
    
    for epoch in range(N_EPOCHS):
        try:
            train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, scheduler)
            f1 = evaluate(model, test_loader)
            
            # --- OPTUNA PRUNING ---
            # Se il trial sta andando male rispetto alla media degli altri, 
            # lo interrompiamo subito per risparmiare tempo e risorse.
            trial.report(f1, epoch)
            if trial.should_prune(): 
                raise optuna.exceptions.TrialPruned()
            
            early_stopper(f1, model)
            best_f1 = max(best_f1, f1)
            
            if early_stopper.early_stop:
                break
                
        except RuntimeError as e:

            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                # Se il batch size era troppo grande, "potiamo" il trial invece di crashare
                raise optuna.exceptions.TrialPruned()
            raise e
        
    return best_f1

# ==================================================================================
# 15. MAIN EXECUTION PIPELINE
# ==================================================================================

def run_experiments():
    """
    Funzione principale che orchestra l'intero benchmark.
    Itera su: Lingue -> Task (A/B) -> Dataset (Originale/Augmented) -> Preprocessing (V1/V2/V3).
    """
    print(f"\n{'='*60}\nAVVIO PIPELINE SPERIMENTALE COMPLETA (IT, EN, ES, MULTI)\n{'='*60}")
    
    global CURRENT_CLEANING_PARAMS, CURRENT_LANG
    results_matrix = []
    
    # 1. Ciclo sulle Lingue
    for lang_code, config in LANG_CONFIG.items():
        CURRENT_LANG = lang_code
        model_name = config['model']
        is_multi = config['is_multi']
        
        print(f"\n\n LINGUA: {lang_code} | BACKBONE: {model_name}")
        
        # Selezione dei file di input corretti
        if is_multi:
            current_datasets = MULTI_DATASETS_PATHS
        else:
            prefix = config['file_prefix']
            current_datasets = {
                '1_Original': BASE_PATH / f'{prefix}.csv',
                '2_Aug_Full_A': BASE_PATH / f'{prefix}_aug_A_full.csv',
                '3_Aug_Bal_B': BASE_PATH / f'{prefix}_aug_B_balanced.csv',
            }
        
        # 2. Ciclo sui Task (A = Testo, B = Testo + Bio)
        for task in ['A', 'B']:

            # Il task B non va eseguito sull inglese
            if lang_code == 'EN' and task == 'B':
                print(f"\nTask B disabilitato per la lingua EN.")
                continue

            # 3. Ciclo sulle varianti di Dataset (Originale vs Augmentations)
            for ds_name, ds_path in current_datasets.items():
                # 4. Ciclo sul Preprocessing (: Emoji vs No-Emoji)
                for prep_name, prep_params in PREPROCESSING_VARIANTS.items():
                    
                    print(f"\nEsecuzione: {lang_code} | Task[{task}] | Data[{ds_name}] | Prep[{prep_name}]")
                    CURRENT_CLEANING_PARAMS = prep_params
                    
                    # Caricamento e Split Dati
                    splits = get_dataset_splits(ds_path, task)
                    if not splits:
                        print(f"Skip: Dataset {ds_name} non trovato o vuoto.")
                        continue
                        
                    X_tr, X_te, y_tr, y_te, weights = splits
                    train_ds, test_ds = create_torch_datasets(X_tr, X_te, y_tr, y_te, model_name)
                    
                    # --- AVVIO OPTIMIZATION (OPTUNA) ---
                    # Usiamo 'MedianPruner': Interrompe un trial se performa peggio 
                    # della mediana dei trial precedenti allo stesso step.
                    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
                    try:
                        study.optimize(
                            lambda t: objective(t, train_ds, test_ds, weights, model_name), 
                            n_trials=N_TRIALS,
                            gc_after_trial=True # Chiama il Garbage Collector dopo ogni trial
                        )
                        
                        bp = study.best_params
                        print(f"Best F1: {study.best_value:.4f}")
                        
                        # Salvataggio Risultati
                        results_matrix.append({
                            'Language': lang_code,
                            'Task': task,
                            'Dataset': ds_name,
                            'Preprocessing': prep_name,
                            'Preprocessing_Details': str(prep_params),
                            'Best_F1_Macro': study.best_value,
                            'Best_Pooling': bp['pooling'],           
                            'Best_Freeze': bp['freeze_layers'],
                            'Best_Optim': bp.get('optimizer', 'AdamW'),
                            'Best_Params': str(bp)
                        })
                        
                    except Exception as e:
                        print(f"Errore critico nel trial: {e}")
                    
                
                    del train_ds, test_ds, study, splits
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Salvataggio incrementale (Checkpointing dei risultati)
                    if results_matrix:
                        pd.DataFrame(results_matrix).to_csv(OUTPUT_DIR / 'RESULTS_ALL_LANGS_PARTIAL.csv', index=False)

    print("\n\n" + "="*60)
    print("ESPERIMENTI COMPLETATI CON SUCCESSO")
    print("="*60)
    
    # Output Finale
    final_df = pd.DataFrame(results_matrix)
    if not final_df.empty:
        # Ordiniamo per performance decrescente
        final_df = final_df.sort_values(by=['Language', 'Task', 'Best_F1_Macro'], ascending=[True, True, False])
        
        cols_to_show = ['Language', 'Task', 'Dataset', 'Preprocessing', 'Best_F1_Macro', 'Best_Pooling']
        print(final_df[cols_to_show])
        final_df.to_csv(OUTPUT_DIR / 'RESULTS_ALL_LANGS_FINAL.csv', index=False)

if __name__ == "__main__":
    run_experiments()
    
