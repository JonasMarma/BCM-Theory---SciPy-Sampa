#coding: UTF-8

import numpy as np
import matplotlib.pyplot as plt
import Image
import pandas as pd
import time

plt.style.use('ggplot')

class receptiveField:
	def __init__(self):
		self.weights = np.random.normal(0.5, 0.15, 512)
		self.weights[self.weights < 0] = 0
		self.theta = 5
		self.yIntegral = 0

		self.imageX = 0
		self.imageY = 0

		self.Npatches = 0
		self.patchSize = 0


	def loadImage(self, name):
		image = Image.open(name)
		image = np.array(image, dtype = "float")
		self.imageX = image.shape[0]
		self.imageY = image.shape[1]
		return image


	def normalize(self, image):
		mean = np.mean(image)
		std = np.std(image)
		image -= mean
		image /= std
		return image


	def createInputs(self, image, xOverlay = 4, yOverlay = 4, patchSize = 16):
		patches = []
		self.patchSize = patchSize

		i = 0; j = 0
		while True:
			if i + patchSize > self.imageX:
				i = 0
				j += patchSize - yOverlay
			if j + patchSize > self.imageY:
				break

			#"Extrair" o quadrado
			patch = image[i:(i+patchSize) , j:(j+patchSize)].flatten()

			#preencher uma linha com o quadrado separando positivos e negativos:
			#1ª metade da linha
			patchHead = (patch > 0) * patch
			#2ª metade da linha
			patchTail = (patch < 0) * np.abs(patch)

			line = np.concatenate((patchHead, patchTail), axis = 0)

			patches.append(line)

			i += (patchSize - xOverlay)

		#Armazenar a quantidade de quadrados extraídos
		self.Npatches = len(patches)

		patches = np.array(patches)

		return patches


	def sigmoid(self, x):
		#print("processing {}".format(x))
		return 1.0/(1.0+np.exp(-x))


	def bcmTraining94(self, inputVect, iteration):
		#Calcular a resposta
		y = self.sigmoid(np.inner(self.weights, inputVect))

		#Alterar os pesos
		d_weights = (inputVect * (y ** 2 - y * self.theta))/self.theta
		self.weights += d_weights
		self.weights[self.weights < 0] = 0

		#Alterar o theta
		self.yIntegral += y
		self.theta = (self.yIntegral/(iteration+1))**2

	def showWeights(self, weights):
		rcpField = []
		rcpField.append(np.array(weights[0:256] - weights[256:512]))

		fig = plt.figure()
		ax1 = fig.add_subplot(111)

		im1 = ax1.imshow(rcpField[0].reshape([self.patchSize,self.patchSize]), cmap='hot')

		cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		plt.colorbar(im1, cax = cbar_ax)

		fig.show()
		plt.pause(100)

	def main(self):
		print("Digite a imagem que deseja processar:")
		img = raw_input()
		image = self.loadImage("images/" + img + ".bmp")
		normImage = self.normalize(image)
		inputVects = self.createInputs(normImage)

		iterations = 150000
		for i in range(0, iterations):
			#Pegando um quadrado aleatório
			randomPatch = np.random.randint(0, self.Npatches, 1)

			self.bcmTraining94(inputVects[randomPatch, :].flatten(), i)

		self.showWeights(self.weights)

rcpFieldInstance = receptiveField()
rcpFieldInstance.main()
