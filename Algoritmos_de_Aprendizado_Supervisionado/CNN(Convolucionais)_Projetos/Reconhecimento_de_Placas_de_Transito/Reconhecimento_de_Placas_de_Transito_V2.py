#Semelhante ao algoritmo utilizado no meu TCC: http://ri.avantis.edu.br/obra/view/92 porém este arquivo aqui nele
#podemos adicionar uma imagem na ultima função e pedir para este algoritmo de Deep Learning verificar se é uma imagem de
#Uma placa Pare ou uma placa Proibido Estaciona, e qual a % de chance para cada caso.
#Estou citando o exemplo com duas placas de transito, mas pode ser quais quer outras duas imagem, isso vai depender
#apenas das imagens tulizads nos dataset de treino e de testes.


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation= 'relu'))  #64, 64, = resoulucao da imagem... 3 significa que é RGB
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation= 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Flatten())

classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=1, activation='sigmoid'))

classificador.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])
gerador_treinamento = ImageDataGenerator(rescale = 1./255,
                                         rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2) #
gerador_teste = ImageDataGenerator(rescale=1./255)

base_treinamento = gerador_treinamento.flow_from_directory('dataset/treino', #dataset utilizado para treinar o o algoritmo de Deep Lerning
                                                           target_size=(64,64),
                                                           batch_size= 32,
                                                           class_mode='binary')

#criando base de dados de teste
base_teste = gerador_teste.flow_from_directory('dataset/teste' #local do dataset de testes,
                                               target_size= (64, 64),
                                               batch_size= 32,
                                               class_mode='binary') 

#funcao de treinamento
classificador.fit_generator(base_treinamento, steps_per_epoch= 270,
                            epochs=30, validation_data = base_teste,
                            validation_steps= 135)



imagem_teste = image.load_img('dataset/teste/4.jpg',
                              target_size = (64,64))
imagem_teste = image.img_to_array(imagem_teste)
imagem_teste /= 255
imagem_teste = np.expand_dims(imagem_teste, axis = 0)
previsao = classificador.predict(imagem_teste)
previsao = (previsao > 0.5)


base_treinamento.class_indices

#Analisar no console a variavel PREVISAO, se o valor que tiver ao lado dela for [TRUE] Entao e Placa Pare se for [FALSE] entao e Proibido Estacionar

