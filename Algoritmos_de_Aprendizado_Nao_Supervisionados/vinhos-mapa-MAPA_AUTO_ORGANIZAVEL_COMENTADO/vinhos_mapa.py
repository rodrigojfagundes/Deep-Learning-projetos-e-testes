from minisom import MiniSom #biblioteca para trabalhar com mapas auto organizaveis
import pandas as pd #importação do pandas

base = pd.read_csv('wines.csv') #importacao da base da dados
X = base.iloc[:,1:14].values #variavel X é todas as colunas menos a coluna CLASSE (POIS ELA E A COLUNA 0)
y = base.iloc[:,0].values #aqui é todas as linhas, e queremos apenas o atributo 0 que é a CLASSE

from sklearn.preprocessing import MinMaxScaler # para fazer a normalização para converter de nome para numeros
normalizador = MinMaxScaler(feature_range = (0,1)) #normalização para converter de nome para numeros
X = normalizador.fit_transform(X) #convetendo os dados entre 0 e 1

#construção do mapa auto organizavel

som = MiniSom(x = 8, y = 8, input_len = 13, sigma = 1.0, learning_rate = 0.5, random_seed = 2) # X = quantas linhas o mapa vai ter, Y quantidade de colunas que o mapa tera, input_len = a quantidade de entrada que teremos ou seja a quantidade de atributos, sigma o valor pode ser = a 1 porque é um valor default o alcance do raio... learning_rate = a taxa de aprendizagem... random_seed = serve para ter sempre o mesmo resultado toda vez que executar, referente a inicialização dos pesos...
som.random_weights_init(X) #inicializando, poem o X que e a base de dados
som.train_random(data = X, num_iteration = 100) #parametro data e o X pois e a base de dados, 100 é o numero de repeticoes

som._weights #visualizar informações como dimensoes e os blocos
som._activation_map #mostra a quantidade de linhas e colunas... mostra os valores do mapa auto organizavel
q = som.activation_response(X) #mostra quantas vezes cada neuronio foi selecionado com o BMU = Best Metodh Unit... Ou seja o neuronio principal, o neuronio mais proximo

from pylab import pcolor, colorbar, plot #
pcolor(som.distance_map().T) # retorna uma matriz com os valores de distancia, com uma matriz transposta(o que e linha vira coluna, e o que é coluna vira linha)
# MID - mean inter neuron distance
colorbar() #desenho da distancia do mapa auto organizavel ao executar essa linha, e com escala... Em que quanto mais escuro mais parecido com os seus vizinhos ele e

w = som.winner(X[2]) #vai dizer qual é o neuronio ganhador de cada um dos registros
markers = ['o', 's', 'D'] 
color = ['r', 'g', 'b'] #separar por cores as classes... 0 bolinha  ...  1 quadradinho ... 2 outro quadradinho
#y[y == 1] = 0 # 1 fica 0
#y[y == 2] = 1 # 2 fica 1
#y[y == 3] = 2 # 3 fica 2

for i, x in enumerate(X): #for para pecorrer todos os registros... traz o ID do registro e a linha completa
    #print(i)
    #print(x)
    w = som.winner(x)
    #print(w)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None', markersize = 10,
         markeredgecolor = color[y[i]], markeredgewidth = 2)
