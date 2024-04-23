import json
import os
import socket
import threading
from util import *
import pickle
import threshold_crypto as tc
from threshold_crypto.data import *
import numpy as np
from numpy import array


c_sockets = []
aggregated_model = {
    "dW": 0,
    "db":0,
    "size":0
}

clients=0
lock = threading.Lock()

def thresh_keys(t,n):
    curve_params = tc.CurveParameters()
    thresh_params = tc.ThresholdParameters(t, n)
    pub_key, key_shares = tc.create_public_key_and_shares_centralized(curve_params, thresh_params)
    for i in range(len(key_shares)):
        key_shares[i] = key_shares[i].to_json()
    return {'pk':pub_key.to_json(),'sks':key_shares, 'thresh_params':thresh_params.to_json()}

def handler(c_socket , wr, br):
    global clients,c_sockets
    with lock:
        c_sockets.append(c_socket)

    # wait until number of clients is N
    print(f'connected clients {len(c_sockets)}',end = '\r')
    if len(c_sockets) == 4:
        FL(wr,br)
    else:
        return 


def FL(wr,br):
    global c_sockets
    epochs = 0
    while epochs < 4:
        print(f"round : {epochs+1}")
        pool = []
        thresh_cntx = thresh_keys(2,4)
        pk = thresh_cntx["pk"]
        thresh_params = thresh_cntx.get('thresh_params')
        sks = thresh_cntx.get('sks')
        
        for i,s in enumerate(c_sockets):
            with lock:
                keys = {
                    'pk':thresh_cntx['pk'],
                    'sk':thresh_cntx['sks'][i],
                    'thresh_params':thresh_cntx['thresh_params']
                }

#
            print(f"sending to client: {i+1}")
            with lock:
                s.send(pickle.dumps(keys))
            print(f"sent to client: {i+1}")
        for s in c_sockets:
            partial_dec = []
            data = EncryptedMessage.from_json(pickle.loads(s.recv(2048)) )
             # data = pickle.loads(s.recv(2048))
            print(data.to_json())
            for j in [0,2,3]:
                partial_decritption = tc.compute_partial_decryption(data,KeyShare.from_json(sks[j]) )
                partial_dec.append(partial_decritption)
            with lock:
                pool.append(eval(tc.decrypt_message(partial_dec,data,ThresholdParameters.from_json(thresh_params))))
            # s.send(b'Hi\n')
            # pool.append(data)

        total_size = sum([c["size"] for c in pool])
        for d in pool:
            aggregated_model["dW"] +=  d["dW"]*d["size"]//total_size
            aggregated_model["db"] +=  d["db"]*d["size"]//total_size
        dWr = aggregated_model["dW"]
        dbr = aggregated_model["db"]
        wr, br = update(dWr, dbr, wr, br, learning_rate = 0.01)
        for s in c_sockets:
            s.send(pickle.dumps([sks[i] for i in [1,2,3]]))
            s.send(pickle.dumps(tc.encrypt_message(str({'W' : wr, 'b' : br}),PublicKey.from_json(pk)).to_json()))
            # s.send(pickle.dumps({'W' : wr, 'b' : br}))
        epochs+=1


def start_server(host,port):
    global clients
    s_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    df = random_dataset("./datasets/data/NF-ToN-IoT.csv",500)
    X_train, _, _, _ = preprocess(df)
    wr, br = initialisation(X_train)
    # bind 
    s_socket.bind((host, port))

    s_socket.listen(4)
    print("Addrese {}:{}".format(host , port))


    try:
        
        while True:
            c_socket , _ = s_socket.accept()
            with lock:
                clients +=1
            print(f"Number of clients: {clients}", end='\r')
            c_thread = threading.Thread(target=handler , args=(c_socket, wr,br ))
            c_thread.start()
    except KeyboardInterrupt:
        s_socket.close()
        print("closing socket")

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 5000
    start_server(host,port)
    print("Fin")
