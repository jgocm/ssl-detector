from cmath import pi
from socket import socket, AF_INET, SOCK_DGRAM
import CommTypes_pb2 as pb
import time

server = "199.0.1.1"
port = 9600

msg = pb.protoPositionSSL()
conn = socket(AF_INET, SOCK_DGRAM)

msg.x = 0
msg.y = 0
msg.w = 0
msg.posType = pb.protoPositionSSL.source
conn.sendto(msg.SerializeToString(), (server, port))

start = time.time()
while True:
    msg.x = 0.75          # radius
    msg.y = 0.95*0.75         # proportion
    msg.w = 0
    msg.posType = pb.protoPositionSSL.source
    msg.front = False
    msg.charge = True   
    msg.chip = False
    #divide por 10 no robo
    msg.kickStrength = 40
    msg.dribbler = False
    msg.dribSpeed = 0
    elapsed_time = time.time() - start
    if elapsed_time > 5:
        msg.charge = False
        msg.front = True
        print("kick")
    if elapsed_time> 30:
        print("stop")
        break 
    #else:
        #print(f'x={msg.x}, y={msg.y}, w={msg.w}')
    conn.sendto(msg.SerializeToString(), (server, port))
    
