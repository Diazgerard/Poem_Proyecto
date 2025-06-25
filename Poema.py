# Leer el archivo
with open("Gutenberg.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Eliminar encabezados/pies del archivo de Gutenberg
start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
if start != -1 and end != -1:
    text = text[start:end]

# Limpiar saltos de línea dobles y espacios extra
text = text.replace('\r', '').replace('\n\n', '\n').strip()
print(f"Longitud del texto: {len(text)} caracteres")

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Crear vocabulario
chars = sorted(set(text))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}

vocab_size = len(chars)
print(f"Vocabulario de {vocab_size} caracteres")

# Convertir texto a enteros
encoded = [char2idx[c] for c in text]


sequence_length = 40
step = 1
X = []
y = []

for i in range(0, len(encoded) - sequence_length, step):
    X.append(encoded[i:i + sequence_length])
    y.append(encoded[i + sequence_length])

X = np.array(X)
y = to_categorical(y, num_classes=vocab_size)

print(f"Total de secuencias: {len(X)}")


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=sequence_length))
model.add(LSTM(128))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.build(input_shape=(None, sequence_length))
model.summary()


model.fit(X, y, batch_size=64, epochs=20)


model.save("poem_generator.h5")

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

def generate_text(seed, length=400, temperature=1.0, force_end_punctuation=True):
    generated = seed
    seed_encoded = [char2idx[c] for c in seed]

    for _ in range(length):
        padded = pad_sequences([seed_encoded[-sequence_length:]], maxlen=sequence_length)
        preds = model.predict(padded, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = idx2char[next_index]
        generated += next_char
        seed_encoded.append(next_index)

    if force_end_punctuation:
        # Buscar desde el final hacia atrás el primer '.', '?', o '!'
        for i in range(len(generated)-1, -1, -1):
            if generated[i] in ['.', '!', '?']:
                generated = generated[:i+1]
                break
        else:
            # Si no se encontró puntuación final, agregamos '.'
            generated += '.'

    return generated


print(generate_text("The sun ", length=400, temperature=1.0))

