
import numpy
from PIL import Image

# FUNCTION DEFINTIONS:

# Abre a imagem e retorna 3 matrizes, cada uma correspondendo a um canal (R, G e B)

def openImage(imagePath):
    imOrig = Image.open(imagePath)
    im = numpy.array(imOrig)

    aRed = im[:, :, 0]
    aGreen = im[:, :, 1]
    aBlue = im[:, :, 2]

    return [aRed, aGreen, aBlue, imOrig]


# Comprime as matrizes

def compressSingleChannel(channelDataMatrix, singularValuesLimit):
    uChannel, sChannel, vhChannel = numpy.linalg.svd(channelDataMatrix)
    aChannelCompressed = numpy.zeros((channelDataMatrix.shape[0], channelDataMatrix.shape[1]))
    k = singularValuesLimit

    leftSide = numpy.matmul(uChannel[:, 0:k], numpy.diag(sChannel)[0:k, 0:k])
    aChannelCompressedInner = numpy.matmul(leftSide, vhChannel[0:k, :])
    aChannelCompressed = aChannelCompressedInner.astype('uint8')
    return aChannelCompressed


# Programa Principal:
print('*** Compressão de Imagem usando SVD ***')
aRed, aGreen, aBlue, originalImage = openImage('image.jpg')

# Dimensões da imagem:
imageWidth = 512
imageHeight = 512

# Número de sigular values para reconstrução da imagem comprimida

singularValuesLimit = 160

aRedCompressed = compressSingleChannel(aRed, singularValuesLimit)
aGreenCompressed = compressSingleChannel(aGreen, singularValuesLimit)
aBlueCompressed = compressSingleChannel(aBlue, singularValuesLimit)

imr = Image.fromarray(aRedCompressed, mode=None)
img = Image.fromarray(aGreenCompressed, mode=None)
imb = Image.fromarray(aBlueCompressed, mode=None)

newImage = Image.merge("RGB", (imr, img, imb))

originalImage.show()
newImage.show()

# Cálculo da taxa de compressão

mr = imageHeight
mc = imageWidth

originalSize = mr * mc * 3
compressedSize = singularValuesLimit * (1 + mr + mc) * 3

print('Tamanho Original:')
print(originalSize)

print('Tamanho comprimido:')
print(compressedSize)

print('Tamanho da taxa de compressão / Tamanho original:')
ratio = compressedSize * 1.0 / originalSize
print(ratio)

print('O tamanho da imagem comprimida é ' + str(round(ratio * 100, 2)) + '% da imagem original ')
print('Finalizado - Imagem comprimida!')