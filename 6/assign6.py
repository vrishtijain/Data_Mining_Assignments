import numpy as np
import sys
import math
import pandas as pd

file = sys.argv[1]
k = int(sys.argv[2])
eps = float(sys.argv[3])
D = pd.read_csv(file , header=None)
D = np.array(D)
D_Y = D[0:, -1]
D = np.delete(D, -1, axis=1)

n , d = D.shape


mean = np.mean(D, axis=0)
# initialise initial means
u_mean_t = np.random.rand(k, d)+mean

cov = np.zeros((k, d, d))
# initializing covariance matrix
for q in range(k):
	for i in range(d):
		for j in range(d):
			if i == j:
				cov[q, i, j] = 1

# initialise prior probability
p_c  =[1/k] * k
# GMM function
def GMM(x, mu, sigma):
	if np.linalg.det(sigma) == 0:
		sig_det = np.linalg.det(sigma+0.01*np.identity(d))
		sig_inv = np.linalg.inv(sigma+0.01*np.identity(d))
	else:
		sig_det = np.linalg.det(sigma)
		sig_inv = np.linalg.inv(sigma)

	z = x-mu
	exp = np.dot(np.dot(z.T, sig_inv), z)
	num = np.exp(-exp/2)
	den = pow(2*math.pi, float(d/2))*pow((sig_det), 1/2)

	return num/den


# define weight matrix
w = np.zeros((k, n))

t = 0
while True:
	t += 1
	
	u_mean = np.copy(u_mean_t)
  # expectation steop
	tot = np.zeros((n))
	for j in range(n):
		for a in range(k):
			tot[j] = tot[j] + GMM(D[j, 0:], u_mean[a, 0:], cov[a, 0:, 0:])*p_c[a]
			

	for i in range(k):
		for j in range(n):
			w[i, j] = (GMM(D[j, 0:], u_mean[i, 0:], cov[i, 0:, 0:])*p_c[i])/tot[j]
			
	 # print(pd.DataFrame(w))
    # maximoization  steop
	for i in range(k):

		 #re estimate mean
		u_mean[i, 0:] = np.dot(D.T, w[i, 0:])/np.sum(w[i, 0:])

	  # re estimate SIGMA
		temp1 = np.zeros((d, d))
		for j in range(n):
			z = D[j, 0:] - u_mean[i]
			z = np.reshape(z, (d, 1))
			temp1 = temp1 + w[i, j]*np.dot(z, z.T)

		cov[i, 0:, 0:] = temp1/np.sum(w[i, 0:])

		# re estimate PROB
		p_c[i] = np.sum(w[i, 0:])/n

	u_mean_t_1 = u_mean

	t1 = 0
	for i in range(k):
		t1 = t1 + pow(np.linalg.norm(u_mean_t_1[i, 0:]-u_mean_t[i, 0:]), 2)

	if t1 <= eps:
		break
	else:
		u_mean_t = u_mean_t_1
		continue


    # print(pd.DataFrame(u_mean_t_1))
    # print(pd.DataFrame(u_mean_t))
print("Final mean of each cluster")
print(pd.DataFrame(u_mean_t_1))
print("Final covariance matrix")
for i in range(k):
    print("\n", pd.DataFrame(cov[i]))
print("\n Number of iteration in EM algo == ", t)

print("Probability")
print(p_c , "\n")

w = w.T
max_index_result = np.argmax(w, axis=1)

print("final cluster of all points according to EM")
print(max_index_result)


print("Final size of each cluster")
unique_elements, counts_elements = np.unique(
	max_index_result, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))


D_Y = pd.DataFrame(D_Y)

unique_class = D_Y[D_Y.columns[0]].unique()

unique_class = list(unique_class)
D_Y = D_Y.T


c_table = np.zeros((k, len(unique_class)))
for i in range(n):
    
    y = unique_class.index(D_Y[i][0])
    x = max_index_result[i]
    c_table[x][y] += 1

c_table = pd.DataFrame(c_table)
c_table.columns = unique_class

print("contigency table ")
print(c_table)



# calculating purity

purity_score = sum (c_table.max(axis=1)) / n
print("purity score == ", purity_score *100, "%")

