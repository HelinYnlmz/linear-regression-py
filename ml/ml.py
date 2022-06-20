import numpy as np

print('Linear regression')
print('\n'+40*'=')

data =\
[
[6.1101,17.592],
[5.5277,9.1302],
[8.5186,13.662],
[7.0032,11.854],
[5.8598,6.8233],
[8.3829,11.886],
[7.4764,4.3483],
[8.5781,12],
[6.4862,6.5987],
[5.0546,3.8166],
[5.7107,3.2522],
[14.164,15.505],
[5.734,3.1551],
[8.4084,7.2258],
[5.6407,0.71618],
[5.3794,3.5129],
[6.3654,5.3048],
[5.1301,0.56077],
[6.4296,3.6518],
[7.0708,5.3893],
[6.1891,3.1386],
[20.27,21.767],
[5.4901,4.263],
[6.3261,5.1875],
[5.5649,3.0825],
[18.945,22.638],
[12.828,13.501],
[10.957,7.0467],
[13.176,14.692],
[22.203,24.147],
[5.2524,-1.22],
[6.5894,5.9966],
[9.2482,12.134],
[5.8918,1.8495],
[8.2111,6.5426],
[7.9334,4.5623],
[8.0959,4.1164],
[5.6063,3.3928],
[12.836,10.117],
[6.3534,5.4974],
[5.4069,0.55657],
[6.8825,3.9115],
[11.708,5.3854],
[5.7737,2.4406],
[7.8247,6.7318],
[7.0931,1.0463],
[5.0702,5.1337],
[5.8014,1.844],
[11.7,8.0043],
[5.5416,1.0179],
[7.5402,6.7504],
[5.3077,1.8396],
[7.4239,4.2885],
[7.6031,4.9981],
[6.3328,1.4233],
[6.3589,-1.4211],
[6.2742,2.4756],
[5.6397,4.6042],
[9.3102,3.9624],
[9.4536,5.4141],
[8.8254,5.1694],
[5.1793,-0.74279],
[21.279,17.929],
[14.908,12.054],
[18.959,17.054],
[7.2182,4.8852],
[8.2951,5.7442],
[10.236,7.7754],
[5.4994,1.0173],
[20.341,20.992],
[10.136,6.6799],
[7.3345,4.0259],
[6.0062,1.2784],
[7.2259,3.3411],
[5.0269,-2.6807],
[6.5479,0.29678],
[7.5386,3.8845],
[5.0365,5.7014],
[10.274,6.7526],
[5.1077,2.0576],
[5.7292,0.47953],
[5.1884,0.20421],
[6.3557,0.67861],
[9.7687,7.5435],
[6.5159,5.3436],
[8.5172,4.2415],
[9.1802,6.7981],
[6.002,0.92695],
[5.5204,0.152],
[5.0594,2.8214],
[5.7077,1.8451],
[7.6366,4.2959],
[5.8707,7.2029],
[5.3054,1.9869],
[8.2934,0.14454],
[13.394,9.0551],
[5.4369,0.61705]
]


X = np.matrix(data)[:,0]

y = np.matrix(data)[:,1]



print('\n y=a*x+b function (a,b=theta)')

def J(X, y, theta):
	theta = np.matrix(theta).T 
	m = len(y) 
	predictions = X * theta 
	sqError = np.power((predictions-y),[2])
	return 1/(2*m) * sum(sqError)


dataX = np.matrix(data)[:,0:1]
X = np.ones((len(dataX),2))
X[:,1:] = dataX

print('\nChecking two example cases of theta:')
for t in [0,0], [-1,2]:
	print('Assuming theta vector at {}, the cost would be {:.2f}'.format(t, J(X, y, t).item()))  



def gradient(X, y, alpha, theta, iters):
	J_history = np.zeros(iters) 
	m = len(y) 
	theta = np.matrix(theta).T 
	for i in range(iters):
		h0 = X * theta 
		delta = (1 / m) * (X.T * h0 - X.T * y) 
		theta = theta - alpha * delta 
		J_history[i] = J(X, y, theta.T) 
	return J_history, theta 
print('\n'+40*'=')

theta = np.matrix([np.random.random(),np.random.random()]) 
alpha = 0.01 
iters = 2000 

print('\n== Model summary ==\nLearning rate: {}\nIterations: {}\nInitial theta: {}\nInitial J: {:.2f}\n'.format(alpha, iters, theta, J(X,y,theta).item()))

print('Training the model... ')
J_history, theta_min = gradient(X, y, alpha, theta, iters)
print('Done.')

print('\nFinal theta: {}\nFinal J: {:.2f}'.format(theta_min.T, J(X,y,theta_min.T).item()))



def predict_profit(population):
	pop = population / 10000
	return [1, pop] * theta_min * 10000

p = 50000 + 100000 * np.random.random()
print('\n'+40*'=')
print('\nBased on learned data, predicted profit for a city of population of {:,.0f} is ${:,.2f}.\n'.format(p, predict_profit(p).item()))

p_min = -theta_min[0].item() / theta_min[1].item() * 10000
print('In order for the business to be profitable, it has to be started in a city with population greater than {:,.0f}.'.format(p_min))
print('\n'+40*'=')
