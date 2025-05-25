import random
import pandas as pd
import json


morse_code = {
    'А': '.-',     'Б': '-...',   'В': '.--',    'Г': '--.',
    'Д': '-..',    'Е': '.',      'Ё': '.',      'Ж': '...-',
    'З': '--..',   'И': '..',     'Й': '.---',   'К': '-.-',
    'Л': '.-..',   'М': '--',     'Н': '-.',     'О': '---',
    'П': '.--.',   'Р': '.-.',    'С': '...',    'Т': '-',
    'У': '..-',    'Ф': '..-.',   'Х': '....',   'Ц': '-.-.',
    'Ч': '---.',   'Ш': '----',   'Щ': '--.-',   'Ъ': '--.--',
    'Ы': '-.--',   'Ь': '-..-',   'Э': '..-..',  'Ю': '..--',
    'Я': '.-.-',

    '0': '-----',  '1': '.----',  '2': '..---',  '3': '...--',
    '4': '....-',  '5': '.....',  '6': '-....',  '7': '--...',
    '8': '---..',  '9': '----.',

    '.': '......', ',': '.-.-.-', ':': '---...', '?': '..--..',
    '-': '-....-', '(': '-.--.-', ')': '-.--.-', '"': '.-..-.',
    '/': '-..-.',  '=': '-...-',  '+': '.-.-.',  '!': '--..--',
    ' ': '/'
}


def dataset_create(size):
    # Сохранение словаря
    with open('morse_dict.json', 'w') as f:
        json.dump(morse_code, f)
        
    random.seed(42)
    dataframe_message = pd.DataFrame(columns=['message'])
    
    for i in range(size):
        message = ""
        message += random.choice(list(morse_code.keys())[:-1])
        for _ in range(random.randint(0, 14)):
            # Генерация сообщений длины от 2 до 15 символов
            message += random.choice(list(morse_code.keys()))
        message += random.choice(list(morse_code.keys())[:-1])
        dataframe_message.loc[i] = [message]

    dataframe_message.to_csv('./morse_dataset.csv', index=False)

if __name__ == '__main__':
    dataset_create(10)
    