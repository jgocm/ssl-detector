import socket

#UDP_IP ='172.20.10.2'
UDP_IP ='169.254.8.26'
UDP_PORT = 5005
MESSAGE = bytes('Hello, World!', 'utf-8')

sock = socket.socket(socket.AF_INET,
                    socket.SOCK_DGRAM)

while True:
    sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
    print(MESSAGE)