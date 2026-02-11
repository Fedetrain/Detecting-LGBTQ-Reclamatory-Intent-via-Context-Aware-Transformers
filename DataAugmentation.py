"""
==================================================================================
SCRIPT DI SUPPORTO: DATA AUGMENTATION VIA BACK-TRANSLATION
==================================================================================

DESCRIZIONE:
Questo modulo non serve per il training, ma per la PREPARAZIONE dei dati.
Genera parafrasi sintattiche dei tweet originali utilizzando la tecnica della 
Back-Translation (Traduzione Andata e Ritorno).
"""

import os
import gc
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm # Barra di caricamento per monitorare il progresso
from transformers import MarianMTModel, MarianTokenizer

# ==================================================================================
# 1. CONFIGURAZIONE AMBIENTE SPECIFICA PER AUGMENTATION
# ==================================================================================

# Configurazione specifica per ambiente Google Colab (se necessario)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    # Percorso base su Google Drive
    BASE_PATH = '/content/drive/MyDrive/Colab Notebooks/dataset/'
except ImportError:
    # Percorso locale di fallback se non eseguito su Colab
    BASE_PATH = './dataset/'

# Percorso di input e output
KAGGLE_INPUT_PATH = BASE_PATH
OUTPUT_DIR = 'output'

# Creazione della directory di output se non esiste
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configurazione del dispositivo di calcolo (GPU se disponibile, altrimenti CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo utilizzato per l'augmentation: {device}")

# ==================================================================================
# 2. CONFIGURAZIONE MODELLI DI TRADUZIONE (MARIAN MT)
# ==================================================================================

# Definizione delle coppie di modelli per la Back Translation.
# Usiamo i modelli Helsinki-NLP (MarianMT).

# STRATEGIA PIVOT:
# Per tradurre l'italiano, passiamo dall'inglese (IT -> EN -> IT).
# Per l'inglese, usiamo lo spagnolo come pivot (EN -> ES -> EN).
TRANSLATION_MODELS = {
    'it': {
        'pivot': 'en',
        'fwd': 'Helsinki-NLP/opus-mt-it-en',  # Modello Andata (IT -> EN)
        'bwd': 'Helsinki-NLP/opus-mt-en-it'   # Modello Ritorno (EN -> IT)
    },
    'es': {
        'pivot': 'en',
        'fwd': 'Helsinki-NLP/opus-mt-es-en',
        'bwd': 'Helsinki-NLP/opus-mt-en-es'
    },
    'en': {
        'pivot': 'es',
        'fwd': 'Helsinki-NLP/opus-mt-en-es',
        'bwd': 'Helsinki-NLP/opus-mt-es-en'
    }
}

# ==================================================================================
# 3. FUNZIONI DI TRADUZIONE 
# ==================================================================================

def translate_batch(texts, model_name, device, batch_size=32):

    print(f"Caricamento modello: {model_name}...")
    
    # Inizializzazione tokenizer e modello
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    
    
    if device.type == 'cuda':
        model.half()

    translated_texts = []

    # Iterazione sui batch di testo

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        # Tokenizzazione con padding e troncamento
        # max_length=512 copre abbondantemente la lunghezza di un tweet
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        # Generazione delle traduzioni

        with torch.no_grad():
            translated = model.generate(**inputs)

        # Decodifica dei token in stringhe (skip_special_tokens rimuove [PAD], [EOS])
        decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
        translated_texts.extend(decoded)

    # PULIZIA AGGRESSIVA DELLA MEMORIA
    # Una volta finite le traduzioni, distruggiamo il modello per fare spazio al prossimo.
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return translated_texts

def perform_back_translation(df, lang_key):
    """
    Esegue la pipeline completa di back-translation per un dato DataFrame.
    1. Traduzione Forward (Lingua Originale -> Pivot)
    2. Traduzione Backward (Lingua Pivot -> Lingua Originale)
    """
    texts = df['text'].tolist()
    config = TRANSLATION_MODELS[lang_key]

    print(f"Traduzione Forward ({lang_key} -> {config['pivot']})...")
    pivot_texts = translate_batch(texts, config['fwd'], device)

    print(f"Traduzione Backward ({config['pivot']} -> {lang_key})...")
    back_translated_texts = translate_batch(pivot_texts, config['bwd'], device)

    return back_translated_texts

# ==================================================================================
#4. ESECUZIONE PIPELINE AUGMENTATION
# ==================================================================================

def run_augmentation_pipeline():
    """
    Funzione principale che itera su tutti i dataset linguistici.
    
    Per ogni lingua, crea due varianti di dataset:
    1. Full Augmented (A): Dataset Originale + Dataset Aumentato Completo (Raddoppio dati).
    2. Balanced Augmented (B): Dataset Originale + Aumentato SOLO per la classe minoritaria.
       Utile per bilanciare le classi.
    """
    print("\nAVVIO PROCEDURA DATA AUGMENTATION (Back-Translation)")

    # Mappa dei file di input per ogni lingua
    files = {
        'en': 'train_en.csv',
        'es': 'train_es.csv',
        'it': 'train_it.csv'
    }

    for lang, filename in files.items():
        # Costruzione del percorso file
        input_path = os.path.join(KAGGLE_INPUT_PATH, filename)

        if not os.path.exists(input_path):
            print(f" Attenzione: File non trovato in {input_path}. Salto...")
            continue

        print(f"\nElaborazione Dataset: {lang.upper()} ({filename})")
        original_df = pd.read_csv(input_path)

        # Definizione della label della classe minoritaria (Hate Speech = 1)
        # Assumiamo che la classe 1 sia quella da aumentare nel bilanciamento
        minority_class_label = 1
        print(f" Dimensioni dataset originale: {len(original_df)} righe")

        # Generazione dati sintetici tramite Back Translation
        augmented_texts = perform_back_translation(original_df, lang)

        # Creazione DataFrame temporaneo con i testi tradotti
        aug_df = original_df.copy()
        aug_df['text'] = augmented_texts 

        # --- CREAZIONE DATASET OPZIONE A (AUGMENTATION COMPLETA) ---
        # Concatena tutto e rimuove duplicati esatti (se la traduzione è identica all'originale).
        df_opt_A = pd.concat([original_df, aug_df]).drop_duplicates(subset=['text'])
        path_A = os.path.join(OUTPUT_DIR, f'train_{lang}_aug_A_full.csv')
        df_opt_A.to_csv(path_A, index=False)
        print(f"Salvato Dataset A (Full): {len(df_opt_A)} righe in {path_A}")
        
        # --- CREAZIONE DATASET OPZIONE B (BILANCIAMENTO CLASSE) ---
        # Aggiunge dati sintetici solo alle label 1.
        aug_minority = aug_df[aug_df['label'] == minority_class_label]
        df_opt_B = pd.concat([original_df, aug_minority]).drop_duplicates(subset=['text'])
        path_B = os.path.join(OUTPUT_DIR, f'train_{lang}_aug_B_balanced.csv')
        df_opt_B.to_csv(path_B, index=False)
        print(f"Salvato Dataset B (Balanced): {len(df_opt_B)} righe in {path_B}")

        # Pulizia della memoria RAM tra una lingua e l'altra
        del original_df, aug_df, df_opt_A, df_opt_B, augmented_texts
        gc.collect()

    print(f"\nProcedura completata. I file sono stati salvati nella directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_augmentation_pipeline()