import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import heapq
import random
import os


def random_dataset(src_file,k):
    sample = os.popen(f"./split.sh {src_file} {k}")
    df = pd.read_csv(sample)
    return df


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

def initialisation(X):
    w = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (w, b)
def update(dW, db, W, b, learning_rate = 0.01):
    W = W - learning_rate * dW
    b = b - learning_rate * db
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

def aggregation_function(parameters):
    aggregated_model = {}
    num_clients = len(parameters)
    for key in parameters[0].keys():
        aggregated_model[key] = 0
        for client_params in parameters:
            for key, value in client_params.items():
                aggregated_model[key] += value
        for key in aggregated_model.keys():
            aggregated_model[key] /= num_clients
        return aggregated_model
if __name__ == "__main__":
   pass 
