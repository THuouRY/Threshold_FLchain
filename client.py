from util import *
import socket
import pickle
import numpy as np
from numpy import array
import threshold_crypto as tc
import ast
import threshold_crypto as tc
from threshold_crypto.data import *
#from scipy.stats import watson
#-----------------Data Preprocessing-----------------------#
df = random_dataset("./datasets/data/NF-ToN-IoT.csv",500)

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

        # print(c_socket.recv(1024).decode())
        print("Waiting for keys")
        try:
            da = c_socket.recv(2048)
            da = pickle.loads(da)
        except Exception as e:
            raise e
        
        print("keys arrived")
        # pk,sk,thresh_params = (ThresholdDataClass.from_json(da.get(k))  for k in da.keys())
        if da:
            print("received keys")
        else:
            print("Nooooo")
            
        pk = PublicKey.from_json(da['pk'])
        sk = KeyShare.from_json(da['sk'])
        thresh_params = ThresholdParameters.from_json(da['thresh_params'])

        print(f'received pk  and secret share{pk}: {sk}',end = '\r')
        params = tc.encrypt_message(str(params),pk)
        params = pickle.dumps(params.to_json())
        c_socket.send(params)
        
        print('client params sent')
        print('receiving model params ...')



        sks = pickle.loads(c_socket.recv(1024))
        data =  c_socket.recv(2048)
        if data:
            print("####################")
            global_model_encrypted = EncryptedMessage.from_json(pickle.loads(data)) 
            partial_dec = []
            for i in sks:
                partial_decritption = tc.compute_partial_decryption(global_model_encrypted,KeyShare.from_json(i))
                partial_dec.append(partial_decritption)
            global_model = eval(tc.decrypt_message(partial_dec,global_model_encrypted,thresh_params))
            # global_model = pickle.loads(data)
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
