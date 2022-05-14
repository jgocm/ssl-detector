from socket import socket, AF_INET, SOCK_DGRAM
import CommTypes_pb2 as pb

server = "199.0.1.1"
port = 9600

msg = pb.protoSpeedSSL()
ssl_socket = socket(AF_INET, SOCK_DGRAM)

msg.vx = 0
msg.vy = 0
msg.vw = 0
msg.front = False
msg.chip = False
msg.kickStrength = 0
msg.dribbler = False
msg.dribSpeed = 0

ssl_socket.sendto(msg.SerializeToString(), (server, port))