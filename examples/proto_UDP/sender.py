from socket import socket, AF_INET, SOCK_DGRAM
import CommTypes_pb2 as pb

server = "199.0.1.1"
port = 9600

'''msg = pb.protoSpeedSSL()

msg.vx = 0
msg.vy = 0
msg.vw = 0
msg.front = False
msg.chip = False
msg.kickStrength = 0
msg.dribbler = False
msg.dribSpeed = 0'''

msg = pb.protoPositionSSL()
msg.x = 1   # positivo = frente
msg.y = 0   # positivo = esquerda
msg.w = 0   # positivo = anti-horário

conn = socket(AF_INET, SOCK_DGRAM)
conn.sendto(msg.SerializeToString(), (server, port))