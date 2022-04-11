import socket
import sys

from numpy import False_
import CommTypes_pb2 as pb

class SocketUDP():
    def __init__(
                self,
                host_address='199.0.1.2',
                host_port=9601,
                device_address='199.0.1.1',
                device_port=9600                
                ):
        super(SocketUDP, self).__init__()
        self.host_address = host_address
        self.host_port = host_port
        self.device_address = device_address
        self.device_port = device_port
    
    def setHostUDP(self, address, port):
        self.host_address=address
        self.host_port=port
    
    def setDeviceUDP(self, address,port):
        self.device_address=address
        self.device_port=port

    def sendPosition(self, x, y, w):
        msg = pb.protoPositionSSL()
        msg.x = x   # positivo = frente
        msg.y = y   # positivo = esquerda
        msg.w = w   # positivo = anti-horário

        conn = socket(socket.AF_INET, socket.SOCK_DGRAM)
        conn.sendto(
            msg.SerializeToString(), 
            (self.device_address, self.device_port)
            )
    
    def sendSpeed(self, vx, vy, vw, front=False, chip=False, kick_strength=0, dribbler=False, dribbler_speed=0):
        msg = pb.protoSpeedSSL()
        msg.x = vx   # positivo = frente
        msg.y = vy   # positivo = esquerda
        msg.w = vw   # positivo = anti-horário
        msg.front = front
        msg.chip = chip
        msg.kickStrength = kick_strength
        msg.dribbler = dribbler
        msg.dribSpeed = dribbler_speed

        conn = socket(socket.AF_INET, socket.SOCK_DGRAM)
        conn.sendto(
            msg.SerializeToString(), 
            (self.device_address, self.device_port)
            )

    def recvOdometry(self):
        msg = pb.protoPositionSSL()

        conn = socket(socket.AF_INET, socket.SOCK_DGRAM)
        conn.bind(
            (self.host_address,self.host_port)
        )

        data, address = conn.recvfrom(1024)
        msg.ParseFromString(data)
        x = msg.x
        y = msg.y
        w = msg.w
        return x, y, w


if __name__ == "__main__":
    host_address = "199.0.1.2"
    host_port = 9601
    device_address = "199.0.1.1"
    device_port = 9600

    UDP = SocketUDP(
        host_address=host_address,
        host_port=host_port,
        device_address=device_address,
        device_port=device_port
    )

    x = 1
    y = 0
    w = 0

    print(f"Sending X, Y,  W Position")
    print(f"x = {x}, y = {y}, w = {w}")
    
    UDP.sendPosition(x, y, w)
