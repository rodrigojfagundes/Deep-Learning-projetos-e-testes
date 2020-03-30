import numpy as np
import pandas as pd
import nltk

from keras.preprocessing.text import Tokenizer #vc especifica qual a quantidade de palavras quer no vocabulário, e ele automaticamente cria esse vocabulário
from keras.preprocessing.sequence import pad_sequences #limitar a sequencia de palavras
from sklearn.model_selection import train_test_split #divide o dataset em treino e teste

from keras.preprocessing.text import text_to_word_sequence
import re, os

from keras.models import Model

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Embedding #recebe o vocabulário
from keras.layers import LSTM, Bidirectional #usando LSTM bidirecional, pq o lstm e passado como parametro para o bidirecional
import matplotlib.pyplot as plt #porta grafico funcao de perda e grafico de acuracia
from nltk.corpus import stopwords #serve para remover algumas palavras do vocabulário, semp precisar especificar
from tqdm import tqdm

#nltk.download() - USE ESSE COMANDO SE NAO FUNCIONAR DE PRIMEIRA

seed = 7
np.random.seed(seed)

# O model será exportado para este arquivo
filename = 'model/model_saved.h5' #nome quer quero usar no modelo exportado

epochs = 1

# dimensionalidade do word embedding pré-treinado
word_embedding_dim = 50

# número de amostras a serem utilizadas em cada atualização do gradiente
batch_size = 32

# Reflete a quantidade máxima de palavras que iremos manter no vocabulário
max_fatures = 200

# Dimensão de saída da camada Embedding
embed_dim = 128

# limitamos o tamanho máximo de todas as sentenças
max_sequence_length = 300

#define se sera usado um World Vectors pre treinado ou não
pre_trained_wv = False

#se sera usado ou nao uma LSTM Bidirectional ou nao é definido aqui
bilstm = False

#funcao que serve para remover caracteres inuteis no vocabulario, lembrando que esse vocabulario ta em ingles
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    cleanr = re.compile('<.*?>')

    string = re.sub(r'\d+', '', string)
    string = re.sub(cleanr, '', string)
    string = re.sub("'", '', string)
    string = re.sub(r'\W+', ' ', string)
    string = string.replace('_', '')

#converter os dados para minusculo, os caracteres acredito eu
    return string.strip().lower()

#funcao usada para importar a planilha e extrair os dados dela
def prepare_data(data):
    data = data[['text', 'sentiment']] #diz que na planilha so quer os dados da coluna TEXT e SENTIMENT

    data['text'] = data['text'].apply(lambda x: x.lower()) #quer todas as frases em letra minuscula
    data['text'] = data['text'].apply(lambda x: clean_str(x)) #inicia a funcao para remover 
    data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x))) #e feito a limpeza novamente e removido caractereses desnecessarios, embora nao precise fazer isso aparementemente... Pois isso ja foi feito acima

    stop_words = set(stopwords.words('english')) #remove as stop words
    text = []
    for row in data['text'].values:
        word_list = text_to_word_sequence(row)
        no_stop_words = [w for w in word_list if not w in stop_words]
        no_stop_words = " ".join(no_stop_words)
        text.append(no_stop_words)

    tokenizer = Tokenizer(num_words=max_fatures, split=' ') #gera p nosso vocabulario com base no numero maximo de palavras que nos colocamos nas stop features

    tokenizer.fit_on_texts(text)
    X = tokenizer.texts_to_sequences(text)

    X = pad_sequences(X, maxlen=max_sequence_length) # aqui limitamos o tamanho maximo da sequencia com base no max_sequence_length
    # X = pad_sequences(X)

    word_index = tokenizer.word_index #sao palavras associadas a outras no indice vetorial, ou seja palavras que tem significado parecido
    Y = pd.get_dummies(data['sentiment']).values #funcao do pandas usada para extrair os rotulos da coluna SENTIMENT, transformando o positivo e negativoem referencias numericas como 0 e 1
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42) #dizendo que so sera utilizado 20% dos dados do dataset para teste

    return X_train, X_test, Y_train, Y_test, word_index, tokenizer

#importando a planilha
data = pd.read_excel('./dataset/imdb.xlsx')  # Lembre de instalar o pacote 'xlrd'

X_train, X_test, Y_train, Y_test, word_index, tokenizer = prepare_data(data)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

#caso use um embedding pre treinado, ira executar o codigo a baixo
def load_pre_trained_wv(word_index, num_words, word_embedding_dim): #index de palavras, quantidade de palavras no vocabulario, palavras no embedding 
    embeddings_index = {} # cria um dicionario e armazena nessa variavel
    f = open(os.path.join('./word_embedding', 'glove.6B.{0}d.txt'.format(word_embedding_dim)), encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs # e armazenado as palavras, mas tambem os coeficientes
    f.close()

    print('%s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((num_words, word_embedding_dim)) #cria matriz de valores nulos... As palavras do embedding seram colocadas nessa matriz de valores nulos, essas palavras depende muito das palavras que estam no nosso dataset
    for word, i in word_index.items():
        if i >= max_fatures:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def model():
    if pre_trained_wv is True:
        print("USE PRE TRAINED")
        num_words = min(max_fatures, len(word_index) + 1)
        weights_embedding_matrix = load_pre_trained_wv(word_index, num_words, word_embedding_dim)
        input_shape = (max_sequence_length,)
        model_input = Input(shape=input_shape, name="input", dtype='int32')
        embedding = Embedding( 
            num_words,
            word_embedding_dim,
            input_length=max_sequence_length, #se estiver usando um word embedding pre treinado o comando da linha 150 vai receber um numero de DIM da word embedding
            name="embedding",
            weights=[weights_embedding_matrix], #sao utilizado os pesos do word embedding
            trainable=False)(model_input) #significa que nao quer treinar a word embedding, porque ela ja foi treinada
        if bilstm is True:
            lstm = Bidirectional(LSTM(word_embedding_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm"))(embedding)
        else:
            lstm = LSTM(word_embedding_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm")(embedding)

    else: #caso nao esteja usando um word embedding pre treinado
        input_shape = (max_sequence_length,)  #o shape de entrada permanece o mesmo
        model_input = Input(shape=input_shape, name="input", dtype='int32')
			#camada embedding nao tera parametro de peso, pois nao estamos usando um word embedding pre treinado
        embedding = Embedding(max_fatures, embed_dim, input_length=max_sequence_length, name="embedding")(model_input)

        if bilstm is True:
            lstm = Bidirectional(LSTM(embed_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm"))(embedding)
        else:
            lstm = LSTM(embed_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm")(embedding)

    model_output = Dense(2, activation='softmax', name="softmax")(lstm) #esse e a nossa saida
    model = Model(inputs=model_input, outputs=model_output)
    return model


model = model()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

if not os.path.exists('./{}'.format(filename)): #o algoritmo vai verificar se tem algum modelo salvo com
#o nome model_saved_lstm.h5, caso sim ele vai usar o modelo, caso nao ele vai criar um arquivocom esse nome

    hist = model.fit(
        X_train,
        Y_train,
        validation_data=(X_test, Y_test), #funcoes que vao ser utilizadas para avaliar a precissao com algoritmo
        epochs=epochs, #numero de epocas, ou seja quantidade de vezes que o sera feito o processo
        batch_size=batch_size,
        shuffle=True,
        verbose=1)

    model.save_weights(filename)

    # Plot graficos
    plt.figure()
    plt.plot(hist.history['loss'], lw=2.0, color='b', label='train')
    plt.plot(hist.history['val_loss'], lw=2.0, color='r', label='val')
    plt.title('Classificador de sentimentos')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy')
    plt.legend(loc='upper right')
    plt.show()

    plt.figure()
    plt.plot(hist.history['acc'], lw=2.0, color='b', label='train')
    plt.plot(hist.history['val_acc'], lw=2.0, color='r', label='val')
    plt.title('Classificador de sentimentos')
    plt.xlabel('Epochs')
    plt.ylabel('Acurácia')
    plt.legend(loc='upper left')
    plt.show()

else:
    model.load_weights('./{}'.format(filename))

scores = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print("Acc: %.2f%%" % (scores[1] * 100))

while True:
    sentence = input("input> ") #Esse e o local que nos inserimos a palavra para o modelo prever se e positiva ou negativa a frase

    if sentence == "exit":
        break

    new_text = [sentence]
    new_text = tokenizer.texts_to_sequences(new_text)

    new_text = pad_sequences(new_text, maxlen=max_sequence_length, dtype='int32', value=0)

    sentiment = model.predict(new_text, batch_size=1, verbose=2)[0]

    if (np.argmax(sentiment) == 0): #a funcao vai verificar o valor mais alto no vetor e vai dizer se o sentimento e positivo ou negativo
        pred_proba = "%.2f%%" % (sentiment[0] * 100)
        print("negativo => ", pred_proba)
    elif (np.argmax(sentiment) == 1):
        pred_proba = "%.2f%%" % (sentiment[1] * 100)
        print("positivo => ", pred_proba)

