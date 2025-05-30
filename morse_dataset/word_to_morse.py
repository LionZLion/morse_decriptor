import numpy as np
import random
import matplotlib.pyplot as plt
from morse_dataset.noise import noise_gen
import json

random.seed(42)

with open('morse_dict.json', 'r') as file:
    morse_code = json.load(file)

morse_to_duration = {
    '.': 1,
    '-': 3,
    ' ': 3,
    '/': 7
}

def encrypt(message):
    cipher = ''
    for letter in message:
        if letter != ' ':
            if letter in morse_code:
                cipher += morse_code[letter] + ' '
            else:
                # Если символ не найден в словаре, добавляем пробел
                cipher += '/'
        else:
            if cipher:
                cipher += '/'
    
    if cipher[-1] == '/':
        cipher = cipher[:-1]
        
    return cipher

def encrypt_to_numpy(message, dt, nu, phase=1, fs=48000):
    cipher = encrypt(message).strip()
    signal = []
    for chr in cipher:
        if chr == ' ' or chr == '/':
            signal.append(np.zeros(morse_to_duration[chr] * dt))
        else:
            duration = morse_to_duration[chr] * dt
            tone = np.sin(2 * np.pi * nu * np.arange(duration) / fs)
            signal.append(tone)
            signal.append(np.zeros(dt))
            
    pre_padding = np.zeros(int(phase * fs * random.random()))
    post_padding = np.zeros(int(phase * fs * random.random()))
    total_signal = np.concatenate([pre_padding] + signal + [post_padding])
    
    return total_signal


def morse_generator(message, dt, nu, noise=False, dev=0):
    array_signal = np.array(encrypt_to_numpy(message.upper(), dt, nu))
    if noise:
        noiser = noise_gen(len(array_signal), dev, 'white')
        array_signal += noiser
    return array_signal

if __name__ == '__main__':
    morse_generator('е', 1000, 500, noise=True, dev=0.5)