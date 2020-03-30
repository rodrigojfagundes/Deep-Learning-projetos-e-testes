import cv2 #Importando biblioteca OPEN CV

imagem = cv2.imread('pessoas.jpg') #IMAGEM DE ENTRADA

classificador = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #utilizando o classificador ja treinado... com o nome de HAARSCACADE_FRONTALFACE
imagemcinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) #convertendo a imagem para uma escala de cinza... para otimizar o desempenho

deteccoes = classificador.detectMultiScale(imagemcinza, scaleFactor=1.1, #passando como parametro do dector a imagem em escala de cinza... scaleFactor define o tamanho da imagem...
                                           minNeighbors=5, #numero de vizinho... quantos vizinho cada retangulo candidato (bolding box) pode ter...
                                           minSize=(30,30), #menor objeto a ser detectado
                                           maxSize=(100,100)) #maior objeto a ser detectado

print(deteccoes) #quantidade de detecções de rostos
print(len(deteccoes)) #quantidade minimia de rostos encontrados

for (x, y, l, a) in deteccoes: # montando o BOLDING BOX ( CAIXA LIMITADORA ) aquele retangulo verde ao redor do rosto
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0,255,0), 2) #fazendo um retangulo na imagem inicial...

cv2.imshow('Detector de faces', imagem) #titulo do quadrado onde aparece a imagem
cv2.waitKey(0) #vai fechar a janela ao apertar qualquer tecla
cv2.destroyAllWindows() #fechar a janela
