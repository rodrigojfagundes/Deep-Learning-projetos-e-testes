from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB


def exibir_resultado(valor):
    comentario, resultado = valor
    resultado = "1 - Comentario positivo" if resultado[0] == '1' else " 0 - Comentario negativo"
    print(comentario, ":", resultado)


def analisar_comentario(classificador, vetorizador, comentario):
    return comentario, classificador.predict(vetorizador.transform([comentario]))


def obter_arq_entrada_das_fontes():
    dir_raiz = "Datasets/"

    with open(dir_raiz + "imdb_labelled.txt", "r") as arquivo_texto:
        arq_entrada = arquivo_texto.read().split('\n')

    with open(dir_raiz + "yelp_labelled.txt", "r") as arquivo_texto:
        arq_entrada += arquivo_texto.read().split('\n')

    with open(dir_raiz + "amazon_cells_labelled.txt", "r") as arquivo_texto:
        arq_entrada += arquivo_texto.read().split('\n')

    return arq_entrada


def tratamento_dos_arq_entrada(arq_entrada):
    arq_entrada_tratados = []
    for dado in arq_entrada:
        if len(dado.split("\t")) == 2 and dado.split("\t")[1] != "":
            arq_entrada_tratados.append(dado.split("\t"))

    return arq_entrada_tratados


def dividir_arq_entrada_para_treino_e_validacao(arq_entrada):
    quantidade_total = len(arq_entrada)
    percentual_para_treino = 0.99
    treino = []
    validacao = []

    for indice in range(0, quantidade_total):
        if indice < quantidade_total * percentual_para_treino:
            treino.append(arq_entrada[indice])
        else:
            validacao.append(arq_entrada[indice])

    return treino, validacao


def pre_processamento():
    arq_entrada = obter_arq_entrada_das_fontes()
    arq_entrada_tratados = tratamento_dos_arq_entrada(arq_entrada)

    return dividir_arq_entrada_para_treino_e_validacao(arq_entrada_tratados)


def realizar_treinamento(registros_de_treino, vetorizador):
    treino_comentarios = [registro_treino[0] for registro_treino in registros_de_treino]
    treino_respostas = [registro_treino[1] for registro_treino in registros_de_treino]

    treino_comentarios = vetorizador.fit_transform(treino_comentarios)

    return BernoulliNB().fit(treino_comentarios, treino_respostas)
total = 0
for indice in range(0, total):
    resultado_analise = analisar_comentario(classificador, vetorizador, avaliacao_comentarios[indice])

    if resultado[0] == '0':
        verdadeiros_negativos += 1 if avaliacao_respostas[indice] == '0' else 0
        falsos_negativos += 1 if avaliacao_respostas[indice] != '0' else 0
    else:
        verdadeiros_positivos += 1 if avaliacao_respostas[indice] == '1' else 0
        falsos_positivos += 1 if avaliacao_respostas[indice] != '1' else 0


registros_de_treino, registros_para_avaliacao = pre_processamento()
vetorizador = CountVectorizer(binary='true')
classificador = realizar_treinamento(registros_de_treino, vetorizador)

exibir_resultado(analisar_comentario(classificador, vetorizador, "this is the best movie"))
exibir_resultado(analisar_comentario(classificador, vetorizador, "this is the worst movie"))
exibir_resultado(analisar_comentario(classificador, vetorizador, "awesome!"))
exibir_resultado(analisar_comentario(classificador, vetorizador, "10/10"))
exibir_resultado(analisar_comentario(classificador, vetorizador, "so bad"))
exibir_resultado(analisar_comentario(classificador, vetorizador, "you is bad"))
exibir_resultado(analisar_comentario(classificador, vetorizador, "i love dog"))
exibir_resultado(analisar_comentario(classificador, vetorizador, "i love cat"))
exibir_resultado(analisar_comentario(classificador, vetorizador, "i hate family happy"))
exibir_resultado(analisar_comentario(classificador, vetorizador, "girl is thin"))
exibir_resultado(analisar_comentario(classificador, vetorizador, "the boy is fat"))
