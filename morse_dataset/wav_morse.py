from morse_dataset.word_to_morse import morse_generator
import shutil
from scipy.io import wavfile
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import random

random.seed(42)

def wav_save(dt, nu, noise=True, dataset_path='./morse_dataset.csv', fs=48000):
    """
    Args:
        dt (int): время точки в отсчётах. Общее время в секундах dt/fs
        nu (int): частота для кода морзе
        dataset_path (str, optional): путь до csv со словами. Defaults to './morse_dataset.csv'.
        fs (int, optional): _description_. Defaults to 48000.

    Returns:
        train_df: исходный датафрейм + путь к айдио файлам
    """
    
    train_df = pd.read_csv(dataset_path)
    if os.path.exists("./signals"):
        shutil.rmtree("./signals")
    os.makedirs("./signals")

    os.makedirs("signals", exist_ok=True)


    for idx, msg in tqdm(enumerate(train_df['message']), total=len(train_df)):
        # Добавление шума
        noise_deviation = [1.0, 1.5, 2, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6]
        dev = random.choice(noise_deviation)
        signal = morse_generator(msg, dt, nu, noise=noise, dev=dev)
        signal = np.float32(signal / np.max(np.abs(signal)))
        
        train_df.loc[idx, 'signal'] = f"./signals/{idx}.wav"
        wavfile.write(f"./signals/{idx}.wav", fs, signal)
    
    return train_df

if __name__ == '__main__':
    wav_save(480, 500, dataset_path='./morse_dataset.csv', fs=48000)