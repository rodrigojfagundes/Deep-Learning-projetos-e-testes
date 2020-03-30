from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM #para implementar a estrutura da rede neural
from sklearn.preprocessing import  MinMaxScaler #para fazer a normalizacao da base de dados
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv('petr4-treinamento.csv') #carregamento base de dados de treinamento
base = base.dropna() #funcao que serve para deletar valores nulos
base_treinamento = base.iloc[:, 1:2].values #seleciona quais sao as colunas que seram utilizadas para fazer treinamento, pega somente o open
base_valor_maximo = base.iloc[:, 2:3].values #aqui vai pegar o valor da alta, do valor mais alto da acao

normalizador = MinMaxScaler(feature_range=(0,1)) #normaliza os dados entre 0 e 1
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)  #normalizacao para colocar os valores da base de dados em uma escala de 0 ate 1
base_valor_maximo_normalizada = normalizador.fit_transform(base_valor_maximo) #normaliza o valor da base valor maximo

previsores = [] #variavel que ira armazenar os90 valores que ficou as acoes da empresa
preco_real1 = [] #preco de abertura
preco_real2 = [] #preco mais alto que a acao foi negociada no dia
for i in range (90, 1242): #pecorrimento 90, ou seja vai começar do registro 90 para baixo... Os 90 registros anteriores ou seja os 90 dias anteriores... 1242 é o total de registro da base de dados
    previsores.append(base_treinamento_normalizada[i-90:i, 0]) #do registro 90 ate o registro 0... ,0 pois so vai usar o open como atributo previsor
    preco_real1.append(base_treinamento_normalizada[i, 0])
    preco_real2.append(base_valor_maximo_normalizada[i, 0])
previsores, preco_real1, preco_real2 = np.array(previsores), np.array(preco_real1), np.array(preco_real2) #transformar dados para ficarem do tipo numpy
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1)) #somente um atributo previsor

preco_real = np.column_stack((preco_real1, preco_real2)) #juntando o preco real1 com o preco real2 - para nao dar erro quando passar como parametro

regressor = Sequential()
regressor.add(LSTM(units=100, return_sequences= True, input_shape=(previsores.shape[1],1))) #units = 100 é o numero de celula de memoria(neuronios). return_sequences = true, é usado apenas quando tem mais de uma camada na rede neural LSTM, pq significa que ele vai passar as inforacoes dessa camada para as proximas camadas. input_shape  {previsores.shape} é o arquivo normalizado com os dados e o [1], [1] é porque so tem um atributo previsor 
regressor.add(Dropout(0.3))#significa que vai zerar 30% das entradas para prevenir o overfitting

regressor.add(LSTM(units=50,return_sequences=True)) #segunda camada com units 50 ou seja 50 neuronios
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50,return_sequences=True)) #terceira camada com units 50 ou seja 50 neuronios
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50))#penultima camada, depois dela e a camada de saida, por isso nao tem a funcao return_sequences
regressor.add(Dropout(0.3))

regressor.add(Dense(units=2, activation='linear')) #camada de saida com 2 neuronios units=2, porque esse algoritmo é de multiplas previsoes de saida

regressor.compile(optimizer='rmsprop', loss='mean_squared_error', #mean_squared_error esta sendo usado para o calculo de erro e ajuste de pesos... 
                  metrics = ['mean_absolute_error']) #O metrics com mean_absolute_error para visualizacao dos resultados
regressor.fit(previsores, preco_real, epochs = 10, batch_size=32) #padrao para EPOCHS e 100, para repetir todo o processo 100 vezes é assim se adaptar melhor aos dados

base_teste = pd.read_csv('petr4-teste.csv') #carregar a base de dados de teste
preco_real_teste = base_teste.iloc[:, 1:2].values #extrair apenas todas as linhas da primeira coluna

base_teste = pd.read_csv('petr4-teste.csv') #carregar a base de dados de teste
preco_real_open = base_teste.iloc[:, 1:2].values #pega a linha e coluna com o preco de abertura(ou seja o primeiro atributo)
preco_real_high = base_teste.iloc[:, 2:3].values #pega linha e coluna com o preco mais alto do dia

base_completa = pd.concat((base['Open'], base_teste['Open']), axis=0) # concatenar a base_teste com a base_completa
entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values #significa que ele vai comecar a buscar os registros a partir da posicao 1152 da base de dados completa... Pois a posicao 1152 corresponde aos dados do mes de dezembro
entradas = entradas.reshape(-1,1) #o -1 significa que nao vai trabalhar com linhas... E depois o 1 para informar a coluna...
entradas = normalizador.transform(entradas) # normalizar os valores

X_teste = [] #recebe lista vazia
for i in range(90, 112):  #112 porque 90 que é a coluna de entrada mais 22 que é a quantidade de registros da base de teste (que representa o mes de janeiro) dai 90 + 22 = 112
    X_teste.append(entradas[i-90:i, 0]) #i-90 para pegar os valores anteriores... e 0 pq ele so ta pegando a coluna com os valores das acoes
X_teste = np.array(X_teste) #conversao para numpy array
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))

previsoes = regressor.predict(X_teste) #previsoes recebe x_teste, mas ele ta em formato 0 e 1
previsoes = normalizador.inverse_transform(previsoes) #inverte a transformacao de 0 e 1 para formato mais legivel

#plotagem
plt.plot(preco_real_open, color ='red', label = 'Preco abertura real')
plt.plot(preco_real_high, color ='black', label = 'Preco alta real')

plt.plot(previsoes[:, 0], color = 'blue', label = 'Previsao abertura')
plt.plot(previsoes[:, 1], color = 'orange', label = 'Previsao alta')

plt.title('previsao preco das acoes janeiro')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()
