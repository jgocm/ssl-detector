from socket import socket, AF_INET, SOCK_DGRAM
import CommTypes_pb2 as pb
import time

server = "199.0.1.1"
port = 9600

msg = pb.protoPositionSSL()
ssl_socket = socket(AF_INET, SOCK_DGRAM)

msg.x = 0   # positivo = frente
msg.y = 0   # positivo = esquerda
msg.w = 0   # positivo = anti-horário
msg.posType = pb.protoPositionSSL.source

ssl_socket.sendto(msg.SerializeToString(), (server, port))

start = time.time()
while True:
    msg.x = 1
    msg.y = 0
    msg.w = 0
    msg.posType = pb.protoPositionSSL.target
    ssl_socket.sendto(msg.SerializeToString(), (server, port))
    elapsed_time = time.time() - start
    if elapsed_time > 10:
        print('stop')
        break
    else:
        print(f'x = {msg.x}, y = {msg.y}, w = {msg.w}')