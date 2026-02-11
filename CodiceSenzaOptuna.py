"""
EVALITA MultiPRIDE 2026 - Final Training Script
Team: GRUPPETTOZZO
Autori: Federico Traina, Alessandro Santoro, Gabriele Greco

DESCRIZIONE:
Questo script allena i modelli finali utilizzando le migliori configurazioni trovate 
tramite Optuna (iperparametri e dataset).
Sostituisce la ricerca degli iperparametri con un loop di training diretto e genera
report dettagliati sulle metriche di classificazione.
"""

# Import delle librerie standard di Python per gestione file e sistema
import os
import sys
import gc  
import random
import time
import copy  # Serve per fare copie profonde (deepcopy) dei dizionari dei pesi del modello
import numpy as np
import pandas as pd

# Import delle librerie PyTorch per il Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import per la visualizzazione dei grafici
import matplotlib.pyplot as plt

# Import per la gestione e pulizia del testo
import emoji                            # Gestione delle emoji (rimozione o conversione in testo)
import nltk                             # Natural Language Toolkit
from nltk.corpus import stopwords       # Stopwords (parole comuni da rimuovere)
from nltk.tokenize import word_tokenize # Tokenizzatore di base
from pathlib import Path                # Gestione moderna dei percorsi file

# Import di Scikit-Learn per metriche e split dei dati
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight  # Per bilanciare le classi
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix

# Import della libreria Datasets di HuggingFace (ottimizzata per velocità)
from datasets import Dataset

# Import della libreria Transformers (HuggingFace) per i modelli BERT
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    get_linear_schedule_with_warmup, 
    logging as hf_logging
)

# Gestione dipendenza opzionale per ottimizzatori avanzati (Lion)
# Se la libreria non c'è, il codice non crasha ma usa un fallback standard.
try:
    from pytorch_optimizer import RAdam, Lion
except ImportError:
    print("Modulo 'pytorch_optimizer' non trovato. Lion non sarà disponibile (fallback su AdamW).")

# Silenzia i warning di HuggingFace per tenere l'output pulito
hf_logging.set_verbosity_error()



# ==================================================================================
# 1. CONFIGURAZIONE AMBIENTE
# ==================================================================================

# Rilevamento automatico dell'ambiente (Colab o Kaggle) per installare dipendenze mancanti
if 'google.colab' in sys.modules or os.path.exists('/kaggle'):
    os.system('pip install -q emoji pytorch-optimizer')

# Definizione intelligente del percorso dei dataset in base all'ambiente in cui gira lo script
if os.path.exists('/kaggle/input/dataaumentato'):
    BASE_PATH = Path('/kaggle/input/dataaumentato')
elif os.path.exists('/kaggle/input/dataset-multipride'):
    BASE_PATH = Path('/kaggle/input/dataset-multipride')
else:
    BASE_PATH = Path('./dataset') # Path locale di default sul tuo PC

# Creazione della cartella per i risultati se non esiste
OUTPUT_DIR = Path('./final_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Percorso Dati Input: {BASE_PATH}")
print(f"Percorso Output: {OUTPUT_DIR}")

# Iperparametri Globali Fissi (non cambiano tra i vari modelli)
SEED = 42             # Seme per la riproducibilità dei risultati
MAX_LENGTH = 128      # Lunghezza massima della frase (token) per BERT
NUM_WORKERS = 2       # Numero di processi paralleli per caricare i dati (Data Loader)
N_EPOCHS = 10         # Numero massimo di epoche di addestramento
PATIENCE = 3          # Numero di epoche da aspettare prima dell'Early Stopping

# Controllo e download delle risorse NLTK necessarie (stopword e tokenizzatori)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    # Scarica silenziosamente se mancano
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Dispositivo di calcolo configurato: {DEVICE}")



# ==================================================================================
# 2. UTILITIES E CLASSI DI SUPPORTO
# ==================================================================================

# Funzione per fissare il SEED ovunque e garantire che i risultati siano identici a ogni run
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)


class EarlyStopping:
    """
    Classe per fermare il training se il modello smette di migliorare.
    Salva anche i pesi migliori ottenuti durante il training.
    """
    def __init__(self, patience=3, min_delta=0, verbose=False):
        self.patience = patience       # Quante epoche aspettare
        self.min_delta = min_delta     # Miglioramento minimo richiesto
        self.verbose = verbose         # Spiega tutto quello che fa 
        self.counter = 0               # Contatore delle epoche senza miglioramenti
        self.best_score = None         # Miglior punteggio F1 visto finora
        self.early_stop = False        # Flag per dire al loop di fermarsi
        self.best_model_state = None   # Qui salveremo i pesi del modello

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())

        # Se il punteggio non migliora 
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True # Attiva lo stop

        # Se il punteggio migliora
        else:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict()) # Salva i nuovi pesi migliori
            self.counter = 0 # Resetta il contatore

    def load_best_weights(self, model):
        """Ricarica nel modello i pesi migliori salvati."""
        if self.best_model_state:
            model.load_state_dict(self.best_model_state)


# ==================================================================================
# 3. PREPROCESSING E TEXT CLEANING
# ==================================================================================

# Variabili globali usate dalla classe statica TextCleaner per sapere come comportarsi
CURRENT_CLEANING_PARAMS = {}
CURRENT_LANG = 'IT'

# Dizionario con le diverse strategie di preprocessing
PREPROCESSING_VARIANTS = {
    'V1_Minimal': {'remove_emojis': False, 'convert_emojis': False, 'remove_stopwords': False},
    'V2_Standard': {'remove_emojis': False, 'convert_emojis': True, 'remove_stopwords': False},
    'V3_Aggressive': {'remove_emojis': True, 'convert_emojis': False, 'remove_stopwords': True}
}

class TextCleaner:
    # Caricamento statico delle stopwords per efficienza (evita di ricaricarle a ogni riga)
    STOPWORDS_DB = {
        'italian': set(stopwords.words('italian')),
        'english': set(stopwords.words('english')),
        'spanish': set(stopwords.words('spanish')),
    }
    # Unione di tutte le stopword per il task multilingua
    STOPWORDS_DB['multi'] = STOPWORDS_DB['italian'] | STOPWORDS_DB['english'] | STOPWORDS_DB['spanish']


    @staticmethod
    def clean(text):
        """Funzione principale di pulizia applicata a ogni riga del dataframe."""
        text = str(text)
        params = CURRENT_CLEANING_PARAMS 
        lang_key = LANG_CONFIG[CURRENT_LANG]['nltk_lang']
        
        # 1. Lowercasing: converte tutto in minuscolo, questo lo facciamo sempre.
        text = text.lower()
        
        # 2. Gestione Emoji
        if params.get('remove_emojis'):
            # Rimuove le emoji completamente
            text = emoji.replace_emoji(text, replace='')
        elif params.get('convert_emojis'):
            # Converte le emoji in testo (es. :smile:) nella lingua corretta
            try:
                if CURRENT_LANG == 'IT': code = 'it'
                elif CURRENT_LANG == 'ES': code = 'es'
                else: code = 'en'
                text = emoji.demojize(text, delimiters=(" ", " "), language=code)
            except:
                text = emoji.demojize(text, delimiters=(" ", " "))
            
        # 3. Rimozione Stopwords
        if params.get('remove_stopwords'):
            words = word_tokenize(text)
            sw_set = TextCleaner.STOPWORDS_DB.get(lang_key, set())
            text = " ".join([w for w in words if w.lower() not in sw_set])
            
        return text


# ==================================================================================
# 4. ARCHITETTURA MODELLO
# ==================================================================================

class AdvancedBERTClassifier(nn.Module):
    """
    Modello PyTorch personalizzato che avvolge un BERT pre-addestrato.
    Supporta strategie di pooling avanzate (non solo il token [CLS]).
    """
    def __init__(self, model_name, dropout_prob=0.2, freeze_layers=0, pooling_type='cls', mlp_hidden_size=256):
        super().__init__()
        self.pooling_type = pooling_type
        # Carica il modello base (backbone) da HuggingFace
        self.bert = AutoModel.from_pretrained(model_name)
        # Flag per capire se è un modello XLM/RoBERTa (che non usano token_type_ids), mentre BERT lo usa
        self.is_xlm = 'xlm' in model_name.lower() or 'roberta' in model_name.lower()
        
        # Logica per congelare (freeze) i primi N layer del trasformatore
        if freeze_layers > 0:                               
            encoder = getattr(self.bert, 'encoder', None)    
            if encoder and hasattr(encoder, 'layer'):
                # Cicla su tutti i layer del modello
                for i, layer in enumerate(encoder.layer):
                    if i < freeze_layers:                     # Se il layer attuale i è inferiore al numero che voglio congelare 
                        for param in layer.parameters():
                            param.requires_grad = False       # Blocca l'aggiornamento dei pesi, quei layer rimangono uguali a come erano nel modello originale pre addestrato

        # PREPARAZIONE DELLA TESTA                   
        hidden_size = self.bert.config.hidden_size            # Legge la dimensione dell output du BERT
        
        # Calcola la dimensione dell'input per il classificatore finale
        # Se usiamo 'concat', uniamo 3 vettori, quindi la dimensione triplica
        input_dim = hidden_size * 3 if pooling_type == 'concat' else hidden_size
            
        # Testa di classificazione (MLP: Multi-Layer Perceptron)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(input_dim, mlp_hidden_size), # Strato denso intermedio, prende il vettore grande e lo comprime ad una dimensione più piccola
            nn.ReLU(),                             # Attivazione non lineare
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_hidden_size, 1)          # Output finale (logits)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Se il modello è Roberta o XLM che non usano il token_type_ids, o se semplicemente non glieli abbiamo passati, chiama BERT con ID e MASK
        if self.is_xlm or token_type_ids is None:
             outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
             outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        last_hidden = outputs.last_hidden_state     # È un tensore che contiene la rappresentazione matematica di ogni singola parola nella frase (Batch_Size, Lunghezza_Frase, 768)
        
        # Implementazione delle diverse strategie di Pooling, dove rappresentiamo la sequenza di vettori, uno per parola, in un unico vettore che rappresenti l`intera frase
        # CASO A: solo cls
        if self.pooling_type == 'cls':
            # Prende solo il vettore del primo token [CLS], che contiene il senso generale della frase
            pooled = last_hidden[:, 0, :]
        
        # CASO B: Fa la media matematica
        # Prepara la maschera per la moltiplicazione, espandendo le dimensioni per farla combaciare con last_hidden
        elif self.pooling_type == 'mean':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            # Moltiplica i vettori delle parole per la maschera (0 o 1). Le parole di padding che hanno maschera 0 vengono azzerate, poi somma tutto
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)

            # Con clamp conta quante parole vere ci sono
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9) # Evita divisione per zero
            # Calcola la media, divide quindi la somma dei vettori per il numero di parole, questo crea un vettore che rappresenta la media semantica di tutte le parole della frase
            pooled = sum_embeddings / sum_mask
        
        # CASO C: Concatena CLS, Media e Max Pooling
        elif self.pooling_type == 'concat':
            # 1. CLS
            cls_out = last_hidden[:, 0, :]
            # 2. Mean
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_out = sum_embeddings / sum_mask
            # 3. Max
            last_hidden_masked = last_hidden.clone()
            last_hidden_masked[input_mask_expanded == 0] = -1e9 # Maschera padding con valore bassissimo
            max_out = torch.max(last_hidden_masked, 1)[0]
            
            # Unione
            pooled = torch.cat((cls_out, mean_out, max_out), dim=1)
        else:
            pooled = last_hidden[:, 0, :] 

        # Passa il vettore risultante nel classificatore finale
        return self.classifier(pooled)


class FocalLoss(nn.Module):
    """
    Loss Function avanzata (migliore della CrossEntropy classica per dati sbilanciati).
    Dà più peso agli esempi "difficili" da classificare.
    """
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha              # Serve a bilanciare l`importanza tra la classe positiva e negativa
        self.gamma = gamma              # Se Gamma = 0, la formula diventa uguale alla CrossEntropy. Più alto è il valore più ignora le cose facili
        self.pos_weight = pos_weight    # Un alternativa ad alpha

    # Funzione che viene eseguita ad ogni batch, inputs sono i logits, cioè i numeri grezzi usciti dal modello
    def forward(self, inputs, targets):
        # Calcola la Binary Cross Entropy di base
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # Tramite la sigmoide trasformo i logits in probabilità
        probs = torch.sigmoid(inputs)
        targets = targets.type_as(probs)
        # La formula serve per unificare il calcolo delle probabilità sia per i casi 0 che 1
        p_t = targets * probs + (1 - targets) * (1 - probs)
        
        # Calcolo del fattore Alpha (bilanciamento classi)
        # Qui decidiamo quanto pesa l`errore in base alla clase reale
        if self.pos_weight is not None:
            alpha_factor = targets * self.pos_weight + (1 - targets)
        else:
            alpha_factor = targets * self.alpha + (1 - targets) * (1.0 - self.alpha)

        # Formula della Focal Loss
        # Caso Facile: Il modello è sicuro (es. p_t = 0.99). (1 - 0.99) = 0.01.
        # 0.01^2 = 0.0001. L'errore viene moltiplicato per 0.0001 -> Diventa quasi zero. Il modello smette di preoccuparsi di questa frase.
        # Caso Difficile: Il modello sbaglia (es. p_t = 0.1). (1 - 0.1) = 0.9.
        # 0.9^2 = 0.81. L'errore viene moltiplicato per 0.81 -> Resta alto. Il modello è costretto a imparare da questo errore.
        loss = alpha_factor * ((1.0 - p_t) ** self.gamma) * bce
        return loss.mean()

# =================================================================================
# 5. DATASET LOADING E GESTIONE FILE
# ==================================================================================

# Configurazione dei modelli da usare per ogni lingua
LANG_CONFIG = {
    'IT': {'model': "lupobricco/umBERTo_fine-tuned_hate_offensivity", 'nltk_lang': 'italian'},
    'EN': {'model': "bert-base-uncased", 'nltk_lang': 'english'},
    'ES': {'model': "dccuchile/bert-base-spanish-wwm-cased", 'nltk_lang': 'spanish'},
    'MULTI': {'model': "xlm-roberta-base", 'nltk_lang': 'multi'}
}

def create_multilingual_datasets():
    """
    Se il task è MULTI, questa funzione unisce i file CSV delle singole lingue
    in un unico file CSV gigante.
    """
    print("\nControllo Dataset Multilingua...")
    # Mappa dei nomi file e suffissi
    files_map = {
        '1_Original':   ('',                'train_multi.csv'),
        '2_Aug_Full_A': ('_aug_A_full',     'train_multi_aug_A_full.csv'),
        '3_Aug_Bal_B':  ('_aug_B_balanced', 'train_multi_aug_B_balanced.csv')
    }
    langs = ['it', 'en', 'es']
    generated_paths = {}
    
    for key, (suffix, out_name) in files_map.items():
        dfs = []
        for lang in langs:
            fname = f"train_{lang}{suffix}.csv"
            fpath = BASE_PATH / fname
            if fpath.exists():
                d = pd.read_csv(fpath)
                d['lang_source'] = lang # Aggiunge colonna origine
                dfs.append(d)
        
        # Se ha trovato i file, li concatena e salva
        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            out_path = OUTPUT_DIR / out_name
            merged_df.to_csv(out_path, index=False)
            generated_paths[key] = out_path
        else:
            # Fallback
            generated_paths[key] = BASE_PATH / out_name 

    return generated_paths


# Genera i path dei dataset multi
MULTI_DATASETS_PATHS = create_multilingual_datasets()

def get_dataset_splits(csv_path, task_type):
    """
    Funzione cuore del data loading: legge CSV, pulisce, divide in Train/Test e calcola i pesi.
    """
    if not csv_path.exists(): 
        print(f"Errore critico: File dataset non trovato in {csv_path}")
        return None, None, None, None, None

    df = pd.read_csv(csv_path)
    
    # Applica la pulizia del testo definita nella classe TextCleaner
    df['text'] = df['text'].apply(TextCleaner.clean)
    
    # Logica specifica per il Task B che ha una colonna 'bio' (contesto)
    if task_type == 'B':
        if 'bio' not in df.columns: df['bio'] = "" 
        else: df['bio'] = df['bio'].fillna('').astype(str).apply(TextCleaner.clean) 
        # Concatenazione speciale col separatore [SEP]
        df['combined_text'] = df['text'] + " [SEP] CONTESTO: " + df['bio']
    else:
        df['combined_text'] = df['text']

    # Rimozione duplicati per evitare bias
    df.drop_duplicates(subset=['combined_text'], keep='first', inplace=True)
    df = df[df['combined_text'].str.strip().astype(bool)] # Rimuove righe vuote
    
    X_text = df['combined_text']
    y = df['label'].astype(int)
    
    # Split Train/Test (80/20) mantenendo la proporzione delle classi (stratify)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.20, stratify=y, random_state=SEED)
    except ValueError:
        # Fallback se le classi sono troppo poche per stratificare
        X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.20, random_state=SEED)

    # Calcolo dei pesi delle classi per bilanciare la loss function
    # (Dà più peso alla classe minoritaria)
    classes = np.unique(y_train)                                                    # Conta etichette uniche 0,1
    cw = compute_class_weight('balanced', classes=classes, y=y_train)               # cw sarà il peso calcolato tramite la funzione che esegue la seguente formula: (Totale campioni) / (Numero classi x Frequenza Classe)
    return X_train, X_test, y_train, y_test, torch.tensor(cw, dtype=torch.float)    # Questo cw viene passato alla Loss Fuctions. Quando la loss calcola l`errore moltiplicherà l`errore della classe per il peso più alto



def create_torch_datasets(X_tr, X_te, y_tr, y_te, model_name):
    """
    Converte i dataframe pandas in Dataset PyTorch già tokenizzati.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Funzione interna di tokenizzazione
    def tok(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH)
    
    # Creazione dataset HuggingFace
    ds_tr = Dataset.from_pandas(pd.DataFrame({'text': X_tr, 'label': y_tr}))
    ds_te = Dataset.from_pandas(pd.DataFrame({'text': X_te, 'label': y_te}))
    
    # Mappatura tokenizzazione 
    ds_tr = ds_tr.map(tok, batched=True).remove_columns(['text'])
    ds_te = ds_te.map(tok, batched=True).remove_columns(['text'])
    
    # Formattazione per PyTorch
    cols = ['input_ids', 'attention_mask', 'label', 'token_type_ids'] 
    ds_tr.set_format("torch", columns=[c for c in cols if c in ds_tr.column_names])
    ds_te.set_format("torch", columns=[c for c in cols if c in ds_te.column_names])
    return ds_tr, ds_te



# ==================================================================================
# 6. TRAINING ENGINE
# ==================================================================================

def train_one_epoch(model, loader, optimizer, loss_fn, scaler, scheduler):
    """Esegue un'epoca completa di training."""
    model.train() 
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad() # Resetta i gradienti precedenti
        
        # Sposta i dati sulla GPU
        input_ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE).float().unsqueeze(1)
        token_type_ids = batch.get('token_type_ids')
        if token_type_ids is not None: token_type_ids = token_type_ids.to(DEVICE)

        # Mixed Precision Training (per risparmiare memoria GPU e velocizzare)
        # Al posto di usare 32 bit per i numeri ne usiamo 16
        with torch.amp.autocast('cuda'):
            logits = model(input_ids, mask, token_type_ids) # Forward pass
            loss = loss_fn(logits, labels) # Calcolo errore
        
        # Backward pass con Scaler (gestisce la precisione mista)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)      
        # Torniamo a 32 bit
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Evita esplosione gradienti
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler: scheduler.step() # Aggiorna il Learning Rate
        total_loss += loss.item()

        # In sintesi:
        # Comprimi i dati per farli stare nella GPU (16 bit).
        # Ingrandisci gli errori per non perderli (Scale).
        # Calcoli le correzioni.
        # Rimpicciolisci le correzioni alla dimensione giusta (Unscale).
        # Applichi le correzioni al modello.
        
    return total_loss / len(loader)


def evaluate_simple(model, loader, loss_fn):
    """Valutazione veloce per il loop di training (solo Loss e F1 macro)."""
    model.eval()
    total_loss = 0
    preds, labels = [], []
    with torch.no_grad(): 
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels_batch = batch['label'].to(DEVICE).float().unsqueeze(1)
            token_type_ids = batch.get('token_type_ids')
            if token_type_ids is not None: token_type_ids = token_type_ids.to(DEVICE)
            
            with torch.amp.autocast('cuda'): 
                logits = model(input_ids, mask, token_type_ids)
                loss = loss_fn(logits, labels_batch)
            
            total_loss += loss.item()
            # Converte logits in predizioni (0 o 1)
            preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int))
            labels.extend(batch['label'].numpy())
            
    f1 = f1_score(labels, preds, average='macro')
    return total_loss / len(loader), f1



def print_full_report(model, loader, lang, task):
    """
    Stampiamo un report finale dettagliato con Accuracy, F1, Precision, Recall e Matrice di Confusione.
    """
    model.eval()
    preds, labels = [], []
    
    print(f"\nGenerazione Report Dettagliato per {lang} Task {task}...")
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            token_type_ids = batch.get('token_type_ids')
            if token_type_ids is not None: token_type_ids = token_type_ids.to(DEVICE)
            
            with torch.amp.autocast('cuda'): 
                logits = model(input_ids, mask, token_type_ids)
            
            preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int))
            labels.extend(batch['label'].numpy())
    
    # Calcolo Metriche
    acc = accuracy_score(labels, preds)
    
    print(f"\n{'='*60}")
    print(f"REPORT FINALE: {lang} - TASK {task}")
    print(f"{'='*60}")
    print(f"Accuracy Globale: {acc:.4f}")
    print("-" * 60)
    print("Classification Report:")
    print(classification_report(labels, preds, target_names=['Non-Hate', 'Hate'], digits=4))
    print("-" * 60)
    print("Matrice di Confusione:")
    cm = confusion_matrix(labels, preds)
    # Gestione sicura nel caso il test set sia piccolissimo e manchi una classe
    if cm.shape == (2, 2):
        print(f"TN: {cm[0][0]} | FP: {cm[0][1]}")
        print(f"FN: {cm[1][0]} | TP: {cm[1][1]}")
    else:
        print(cm)
    print(f"{'='*60}\n")

# ==================================================================================
# 7. ESECUZIONE MIGLIORI CONFIGURAZIONI
# ==================================================================================

# Elenco dei parametri ottimali trovati tramite il codice che implementa optuna.
BEST_CONFIGS = [
    {
        'Language': 'IT', 'Task': 'A', 
        'Dataset': '2_Aug_Full_A', 'Preprocessing': 'V1_Minimal', 
        'Pooling': 'concat', 'Freeze': 4, 'Optim': 'Lion', 
        'Params': {'lr': 4.44e-05, 'bs': 16, 'dr': 0.37, 'mlp_hidden': 128, 'loss': 'BCE'}
    },
    {
        'Language': 'IT', 'Task': 'B', 
        'Dataset': '2_Aug_Full_A', 'Preprocessing': 'V2_Standard', 
        'Pooling': 'mean', 'Freeze': 4, 'Optim': 'AdamW', 
        'Params': {'lr': 2.21e-05, 'bs': 32, 'dr': 0.12, 'mlp_hidden': 128, 'loss': 'Focal', 'gamma': 2.46}
    },
    {
        'Language': 'EN', 'Task': 'A', 
        'Dataset': '3_Aug_Bal_B', 'Preprocessing': 'V2_Standard', 
        'Pooling': 'concat', 'Freeze': 4, 'Optim': 'Lion', 
        'Params': {'lr': 3.84e-05, 'bs': 16, 'dr': 0.17, 'mlp_hidden': 256, 'loss': 'Focal', 'gamma': 1.04}
    },
    {
        'Language': 'ES', 'Task': 'A', 
        'Dataset': '2_Aug_Full_A', 'Preprocessing': 'V1_Minimal', 
        'Pooling': 'cls', 'Freeze': 6, 'Optim': 'Lion', 
        'Params': {'lr': 4.74e-05, 'bs': 16, 'dr': 0.18, 'mlp_hidden': 128, 'loss': 'Focal', 'gamma': 0.63}
    },
    {
        'Language': 'ES', 'Task': 'B', 
        'Dataset': '2_Aug_Full_A', 'Preprocessing': 'V2_Standard', 
        'Pooling': 'cls', 'Freeze': 0, 'Optim': 'AdamW', 
        'Params': {'lr': 2.30e-05, 'bs': 16, 'dr': 0.26, 'mlp_hidden': 256, 'loss': 'Focal', 'gamma': 2.90}
    },
    {
        'Language': 'MULTI', 'Task': 'A', 
        'Dataset': '3_Aug_Bal_B', 'Preprocessing': 'V2_Standard', 
        'Pooling': 'mean', 'Freeze': 5, 'Optim': 'Lion', 
        'Params': {'lr': 3.53e-05, 'bs': 16, 'dr': 0.37, 'mlp_hidden': 128, 'loss': 'Focal', 'gamma': 2.86}
    },
    {
        'Language': 'MULTI', 'Task': 'B', 
        'Dataset': '3_Aug_Bal_B', 'Preprocessing': 'V2_Standard', 
        'Pooling': 'concat', 'Freeze': 0, 'Optim': 'Lion', 
        'Params': {'lr': 1.18e-05, 'bs': 16, 'dr': 0.30, 'mlp_hidden': 128, 'loss': 'Focal', 'gamma': 1.20}
    }
]

print(f"\nInizio Pipeline di Training per {len(BEST_CONFIGS)} modelli...\n")

# Loop principale: itera su ogni configurazione e addestra un modello
for i, config in enumerate(BEST_CONFIGS):
    lang = config['Language']
    task = config['Task']
    print(f"\n==================================================")
    print(f"MODELLO {i+1}/{len(BEST_CONFIGS)}: {lang} - Task {task}")
    print(f"==================================================")
    print(f"Config: {config['Dataset']} | {config['Preprocessing']} | {config['Pooling']} pool | {config['Optim']}")
    
    # 1. Setup Parametri Preprocessing Globali
    CURRENT_LANG = lang

 
    CURRENT_CLEANING_PARAMS = PREPROCESSING_VARIANTS[config['Preprocessing']]
        
    # 2. Selezione File Dataset corretto
    ds_key = config['Dataset']
    if lang == 'MULTI':
        csv_path = MULTI_DATASETS_PATHS.get(ds_key)
    else:
        if ds_key == '1_Original': suffix = ''
        elif ds_key == '2_Aug_Full_A': suffix = '_aug_A_full'
        elif ds_key == '3_Aug_Bal_B': suffix = '_aug_B_balanced'
        else: suffix = ''
        csv_path = BASE_PATH / f"train_{lang.lower()}{suffix}.csv"
        
    print(f" Dataset: {csv_path}")
    
    # 3. Caricamento Dati e Split
    X_tr, X_te, y_tr, y_te, class_weights = get_dataset_splits(csv_path, task)
    if X_tr is None: 
        print("Skip per mancanza dati.")
        continue
    
    # 4. Creazione Dataset e DataLoader PyTorch
    model_name = LANG_CONFIG[lang]['model']
    train_ds, test_ds = create_torch_datasets(X_tr, X_te, y_tr, y_te, model_name)
    
    bs = config['Params']['bs'] # Batch size
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=NUM_WORKERS)
    
    # 5. Inizializzazione Modello
    model = AdvancedBERTClassifier(
        model_name=model_name,
        dropout_prob=config['Params']['dr'],
        freeze_layers=config['Freeze'],
        pooling_type=config['Pooling'],
        mlp_hidden_size=config['Params']['mlp_hidden']
    ).to(DEVICE)
    
    # 6. Configurazione Ottimizzatore (Lion o AdamW)
    lr = config['Params']['lr']
    optim_name = config['Optim']
    
    if optim_name == 'Lion':
        try:
            optimizer = Lion(model.parameters(), lr=lr, weight_decay=1e-2)
        except NameError:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
    # 7. Configurazione Loss (Focal Loss o BCE)
    if config['Params']['loss'] == 'Focal':
        loss_fn = FocalLoss(gamma=config['Params'].get('gamma', 2.0), pos_weight=class_weights[1].to(DEVICE))
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights[1].to(DEVICE))
        
    # Inizializzazione Scaler per mixed precision 
    scaler = torch.amp.GradScaler('cuda')
    # Scheduler per ridurre il learning rate gradualmente
    total_steps = len(train_loader) * N_EPOCHS
    warmup_steps = int(total_steps * 0.1) 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(train_loader)*N_EPOCHS)
    early_stopper = EarlyStopping(patience=PATIENCE, verbose=True)
    
    # 8. Training Loop Effettivo
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    print("Start Training...")
    start_time = time.time()
    
    for epoch in range(N_EPOCHS):
        t_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, scheduler)
        v_loss, v_f1 = evaluate_simple(model, test_loader, loss_fn)
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_f1'].append(v_f1)
        
        print(f"Ep {epoch+1:02d} | Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val F1: {v_f1:.4f}")
        
        early_stopper(v_f1, model)
        if early_stopper.early_stop:
            print("Early Stopping!")
            break
            
    # 9. Fine Training, Report e Pulizia
    print(f" Tempo: {(time.time() - start_time)/60:.1f} min")
    
    # Ricarica i pesi migliori (quelli con F1 maggiore)
    early_stopper.load_best_weights(model)
    # Stampa metriche complete
    print_full_report(model, test_loader, lang, task)
    
    # Salva il grafico dell'andamento Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f"Training: {lang} Task {task}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plot_path = OUTPUT_DIR / f"plot_{lang}_{task}.png"
    plt.savefig(plot_path)
    plt.close()
    
    # Pulizia manuale memoria GPU 
    del model, optimizer, scaler, scheduler
    torch.cuda.empty_cache()
    gc.collect()

print("\n TUTTI I TRAINING COMPLETATI CON SUCCESSO.")