import os
import socket
import threading
from util import *
import pickle
import threshold_crypto as tc


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
    return (pub_key,key_shares)

def handler(c_socket , wr, br):
    global clients,c_sockets
    c_sockets.append(c_socket)
    
    # wait until number of clients is N
    print(f'connected clients {len(c_sockets)}',end = '\r')
    if len(c_sockets) == 4:
        FL(wr,br)
            

def FL(wr,br):
    global c_sockets
    epochs = 0
    while epochs < 4:
        print(f"round : {epochs+1}")
        pool = []
        pk , sks = thresh_keys(2,4)
        # sending secret shares:
        for i,s in enumerate(c_sockets):
            # s.send(pickle.dumps((pk,sks[i])))
            pool.append(pickle.loads(s.recv(2048)))
        total_size = sum([c["size"] for c in pool])
        for d in pool:
            aggregated_model["dW"] +=  d["dW"]*d["size"]//total_size
            aggregated_model["db"] +=  d["db"]*d["size"]//total_size
        dWr = aggregated_model["dW"]
        dbr = aggregated_model["db"]
        wr, br = update(dWr, dbr, wr, br, learning_rate = 0.01)
        for s in c_sockets:
            s.send(pickle.dumps({'W' : wr, 'b' : br}))
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

    s_socket.listen(6)
    print("Addrese {}:{}".format(host , port))


    try:
        while True:
            c_socket , _ = s_socket.accept()
            clients +=1
            print(f"Number of clients: {clients}", end='\r')
            c_thread = threading.Thread(target=handler , args=(c_socket, wr,br ))
            c_thread.start()
    except Exception as e:
        raise e
    finally:
        s_socket.close()

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 5000
    start_server(host,port)
    print("Fin")
