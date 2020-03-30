from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization #para normalizar os dados
from keras.preprocessing.image import ImageDataGenerator #para gerar algumas imagens adicionais
import numpy as np #sera usado para eu enviar a imagem que eu quero classificar se é gato ou cachorro
from keras.preprocessing import image #utilizado para fazer a leitura da imagem para classificar se e gato ou cachorro

classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation= 'relu')) #primeira camada de convolucao, com um filtro de 32, e as dimensoes de 3, 3(MATRIZ DE 9 PIXELS)64, 64 e a lagura e altura de imagens, o numero 3 significa que sao 3 cores (RGB) ou seja a imagem e colorida... Funcao de ativacao e a relu, q retira as partes escuras de imagem
classificador.add(BatchNormalization()) #acelera o processamento pegando o mapa de caracteristica de imagens que foi gerado pelo o kernel do detector de caracteristicae deixar em escala de 0 e 1
classificador.add(MaxPooling2D(pool_size=(2,2))) #
#mais uma camada de convolucao a baixo
classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation= 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Flatten()) #transformar a matriz em vetor para nos passarmos como entrada para nossa rede neural
#criacao da rede neural densa
classificador.add(Dense(units = 128, activation = 'relu')) #128 numero de neuronios de entrada, relu funcao de ativacao
classificador.add(Dropout(0.2)) #vai zerar 20% das entradas
classificador.add(Dense(units=128, activation='relu')) #mais uma camada oculta
classificador.add(Dropout(0.2)) #zerar alguns neuronios dessa caada tambem
classificador.add(Dense(units=1, activation='sigmoid')) #um unico neuronio de saida, pq esse e um problema de classificacao binaria... Ou seja so tem duas opcoes...

classificador.compile(optimizer='adam', loss='binary_crossentropy', #se tivesse mais de duas opcoes de saida, seria o categorical_crossentropy(no caso e duas gato ou cachorro)
                      metrics=['accuracy'])
gerador_treinamento = ImageDataGenerator(rescale = 1./255, #gera as imagens que seram usadas no treinamento, e faz a normalizacao
                                         rotation_range = 7, #esse parametro indica o grau que sera feito a rotacao na imagem
                                         horizontal_flip = True, #significa que ele vai fazer giros horizontais nas imagens
                                         shear_range = 0.2, #vai fazer mudanca de pixels para outra direcao
                                         height_shift_range = 0.07, #vai fazer a faixa de mudanca da altura
                                         zoom_range = 0.2)
gerador_teste = ImageDataGenerator(rescale=1./255)
#criacao de base da dados de treinamento
base_treinamento = gerador_treinamento.flow_from_directory('dataset/dataset/training_set', #aqui é passado o diretorio onde ta as imagems
                                                           target_size=(64,64), #tamanho das imagens
                                                           batch_size= 32,
                                                           class_mode='binary') #como é duas classes é binario
#criando base de dados de teste
base_teste = gerador_teste.flow_from_directory('dataset/dataset/test_set', #local onde ta a base de teste
                                               target_size= (64, 64),#tamanho das imagens
                                               batch_size= 32,
                                               class_mode='binary') #duas categorias de imagens, entao e binario
#funcao de treinamento
classificador.fit_generator(base_treinamento, steps_per_epoch= 4000 / 2, #recomendaram que é bom dividir por 32 para ser mais rapido
                            epochs=4, validation_data = base_teste, #epochs é 2 para fazer o processo de treinamento 2 vezes... E validar os dados com o s da base_teste
                            validation_steps=1000/2)

imagem_teste = image.load_img('dataset/dataset/test_set/gato/cat.3500.jpg', #colocado o local da imagem que queremos classificar como gato ou cachorro
                              target_size = (64,64)) #tamanho da imagem(eu acho ou filtro)
imagem_teste = image.img_to_array(imagem_teste) #converter a imagem
imagem_teste /= 255 #normalizacao para poros valores na escala de 0 e 1
imagem_teste = np.expand_dims(imagem_teste, axis = 0) #para expandir as dimensoes
previsao = classificador.predict(imagem_teste) #funcao de previsao, que retorna a probabilidade de ser cachorro ou gato
previsao = (previsao > 0.5) # se for a baixo de 0.5 é cachorro e se for acima de 0.5 e gato


base_treinamento.class_indices

#Analisar no conole a variavel PREVISAO, se o valor que tiver ao lado dela for [TRUE] Entao e GATO se for [FALSE] entao e CACHORRO
