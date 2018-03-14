#coding: UTF-8

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

epochs = 3000

plt.style.use('ggplot')

class postsynaptic_neuron:

	def __init__(self):
		self.weights = np.random.uniform(0.0, 1.0, 100)
		self.theta = 2.5
		self.yIntegral = 0

		#Para fazer os gráficos
		self.weightsHistory = np.zeros((epochs, 100))
		self.thetaHistory = []


	def sigmoid(self, x):
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

		#Utilizado para fazer os gráficos
		self.thetaHistory.append(self.theta)
		self.weightsHistory[iteration] = self.weights


	def get_parameters(self):
		return [self.weights, self.thetaHistory, self.weightsHistory]


#===========SETUP============
postneuron = postsynaptic_neuron()

print("Digite as médias das gaussianas:")
s = raw_input()
means = map(int, s.split())
std = 10
input_neurons = np.arange(1,101)

inputs = []
for g in means:
	inputs.append(stats.norm.pdf(input_neurons, g, std))
#===========================

#TREINAMENTO
for i in range(0, epochs):
	randInput = np.random.randint(0, len(means), 1)
	presynaptic = inputs[randInput]
	postneuron.bcmTraining94(presynaptic, i)

	#plt.plot(input_neurons, presynaptic, linewidth=2.5)
	#plt.show()


[final_weights, final_thetaHistory, weights] = postneuron.get_parameters()

fig = plt.figure()
fig.suptitle('Seletividade no campo receptivo', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85, hspace=0.6)
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

ax1.set_title('Pesos finais')
ax2.set_title(u'Evolução do limiar')
ax3.set_title(u'Evolução dos pesos')

ax1.set_xlim(0, 100)
ax2.set_xlim(0, epochs)

ax1.bar(np.arange(0,100),final_weights)
ax2.plot(final_thetaHistory)

weights = np.transpose(weights)

ax3.imshow(weights, cmap=plt.cm.BuPu_r, interpolation='nearest', aspect='auto')
#(np.arange(0, epochs)

fig.show()
plt.pause(1000)
