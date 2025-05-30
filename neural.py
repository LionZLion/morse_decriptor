import torch
import torch.nn as nn
import numpy as np
from morse_dataset.word_to_morse import morse_generator
from CTC_decoder import CTC_decoder
from stft import fft_morse
from model_class import Morse_Decoder


def morse_processing(msg, dict_params, dev=0.5):

    # dt - Число отсчётов. Время точки = dt / fs
    dt = dict_params['dt']
    # Частота
    nu = dict_params['nu']
    # Частота дискретизации
    fs = dict_params['fs']
    step = dict_params['step']
    size = dict_params['size']

    signal = morse_generator(msg, dt, nu, noise=True, dev=dev)
    signal = np.float32(signal / np.max(np.abs(signal)))
    fft_transform = fft_morse(signal, step=step, size=size, nu=nu, fs=fs)
    signal_stft = torch.tensor(fft_transform, dtype=torch.float)

    model = Morse_Decoder(57)
    weights = torch.load('best_model.tar', weights_only=True)
    model.load_state_dict(weights)
    model.eval()

    with torch.no_grad():
        predict = model(signal_stft.unsqueeze(0).unsqueeze(0))
        predict = nn.functional.log_softmax(predict, dim=-1)
        pred_indices = torch.argmax(predict, dim=-1)
        pred_indices = pred_indices.permute(1, 0).squeeze(0)
        main_answer = CTC_decoder(pred_indices)
    
    return main_answer

if __name__ == '__main__':
    msg = 'Привет'
    dict_params = {
        'dt': 3000,
        'nu': 1000,
        'fs': 48000,
        'step': 50,
        'size': 960,
    }
    print(*morse_processing(msg, dict_params, dev=0.5))