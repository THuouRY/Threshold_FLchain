from util import *
import socket
import pickle
import random
import heapq
import numpy as np
from sklearn.model_selection import train_test_split
import threshold_crypto as tc
import ast
#from scipy.stats import watson
#-----------------Data Preprocessing-----------------------#
df = random_dataset("./datasets/data/NF-ToN-IoT.csv",500)

def preprocess(df):
    src_ipv4_idx = {name: idx for idx, name in enumerate(sorted(df["IPV4_SRC_ADDR"].unique()))}
    dst_ipv4_idx = {name: idx for idx, name in enumerate(sorted(df["IPV4_DST_ADDR"].unique()))}

    df["IPV4_SRC_ADDR"] = df["IPV4_SRC_ADDR"].apply(lambda name: src_ipv4_idx[name])
    df["IPV4_DST_ADDR"] = df["IPV4_DST_ADDR"].apply(lambda name: dst_ipv4_idx[name])
    df = df.drop('Attack', axis=1)

    X=df.iloc[:, :-1].values
    y=df.iloc[:, -1].values
    X = (X - X.min()) / (X.max() - X.min())
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (X_train, X_test, y_train, y_test)
#-------------------Apprentissage Machine---------------------------#
def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)
def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A
def log_loss(A, y):
    epsilon = 1e-15
    return 1/len(y) * (np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon)))
def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)
def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)
def predict(X, W, b):
    A = model(X, W, b)
    # print(A)
    return A >= 0.5
def artificial_neuron(W, b, X, y, learning_rate = 0.01, n_iter = 100, batch_size=64):
    Loss = []
    m = X.shape[0]
    for i in range(n_iter):
        for j in range(0, m, batch_size):
            # Get mini-batch
            X_batch = X[j:j+batch_size]
            y_batch = y[j:j+batch_size]

            # Forward propagation
            A = model(X_batch, W, b)
            loss = log_loss(A, y_batch)

            # Backpropagation
            dW, db = gradients(A, X_batch, y_batch)

            # Update parameters
            W, b = update(dW, db, W, b, learning_rate)

        # Compute average loss for the epoch
        epoch_loss = log_loss(model(X, W, b), y)
        Loss.append(epoch_loss)

        print(f"Iteration {i+1}/{n_iter} - Loss: {epoch_loss}")
    #print(Loss)
    # plt.plot(Loss)
    # plt.show()
    return (W, b, dW, db)
def server_model(W, b, dW, db):
    W, b = update(dW, db, W, b, learning_rate = 0.01)
    return (W, b)
def euclidean_distance(v1, v2):
    return np.linalg.norm(np.array(v1) - np.array(v2))
def k_neighbors(v, V, k):
    n = len(V)
    neighbors = []
    for vector in V:
        x_v = random.uniform(0, 0.0001)
        distance = euclidean_distance(v, vector) + x_v
        heapq.heappush(neighbors, (distance, vector))

    k_nearest = heapq.nsmallest(k, neighbors)
    k_nearest_vectors = [vector for _, vector in k_nearest]
    return k_nearest_vectors
def s(v, V, f):
    n = len(V)
    sum = 0
    n_f_2_neighbors = k_neighbors(v, V, n-f-2)
    for vector in n_f_2_neighbors:
        sum += euclidean_distance(v, vector)
    return sum
def krum(V, f):
    n = len(V)
    S = []
    for v in V:
       s_v = s(v, V, f)
       S += [s_v]
    return min(S)
def generate_random(i, n):
    random_number = random.random()
    random_number *= 1/n
    random_number += i/n
    return random_number
def main():
    #----client data prep----#
    X_train, X_test, y_train, y_test = preprocess(df)
    W, b = initialisation(X_train)
    valeur1 = [generate_random(7, 100) for _ in range(7)]
    somme_totale1 = sum(valeur1)
    valeurs_normalisee1 = [valeur / somme_totale1 for valeur in valeur1]
    rounds = 0
    W1, b1 = W, b
    #----client socket----#
    c_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_conf = ('localhost', 5000)
    # client_conf = ('localhost', 12346)
    c_socket.connect(server_conf)
    # c_socket.bind(client_conf)
    #----client training----#
    while rounds < 4:
        print("Round : ", rounds)
        W1, b1, dW1, db1 = artificial_neuron(W1, b1, X_train, y_train, learning_rate = 0.01, n_iter = 40)
        params = {'dW' : dW1, 'db': db1,'size': 500}
        
        # pk,sk = pickle.loads(c_socket.recv(2048))
        # 
        # print(f'received pk  and secret share{pk}: {sk}',end = '\r')
        # params = tc.encrypt_message(str(params),pk)
        params = pickle.dumps(params)
        c_socket.sendall(params)
        print('client params sent')
        print('receiving model params ...')
        data = c_socket.recv(2048)
        if data:
           print("####################")
           global_model = pickle.loads(data)
           print(global_model)
           W1 = global_model['W']
           b1 = global_model['b']
            
        else:
            print('closing the socket')
            c_socket.close()
            break
        rounds += 1
    print("FIN")

        

if __name__ == "__main__":
    main()
