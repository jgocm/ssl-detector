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
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    def setHostUDP(self, address, port):
        self.host_address=address
        self.host_port=port
    
    def setDeviceUDP(self, address,port):
        self.device_address=address
        self.device_port=port

    def sendStopMotion(self):
        msg = pb.protoPositionSSL()

        msg.x = 0   # positivo = frente
        msg.y = 0   # positivo = esquerda
        msg.w = 0   # positivo = anti-horário
        msg.posType = pb.protoPositionSSL.stop

        self.udp_sock.sendto(msg.SerializeToString(), (self.device_address, self.device_port))

    def sendRotateSearch(self, x, y, w):
        msg = pb.protoPositionSSL()

        msg.x = x   # positivo = frente
        msg.y = y   # positivo = esquerda
        msg.w = w   # positivo = anti-horário
        msg.posType = pb.protoPositionSSL.rotateOnSelf

        self.udp_sock.sendto(msg.SerializeToString(), (self.device_address, self.device_port))

    def sendRotateInPoint(self, x, y, w):
        msg = pb.protoPositionSSL()

        msg.x = x   # positivo = frente
        msg.y = y   # positivo = esquerda
        msg.w = w   # positivo = anti-horário
        msg.posType = pb.protoPositionSSL.rotateInPoint

        self.udp_sock.sendto(msg.SerializeToString(), (self.device_address, self.device_port))


    def sendRotateControl(self, x, y, w):
        msg = pb.protoPositionSSL()

        msg.x = x   # positivo = frente
        msg.y = y   # positivo = esquerda
        msg.w = w   # positivo = anti-horário
        msg.posType = pb.protoPositionSSL.rotateControl

        self.udp_sock.sendto(msg.SerializeToString(), (self.device_address, self.device_port))

    def sendBallDocking(self, x, y, w):
        msg = pb.protoPositionSSL()

        msg.x = x   # positivo = frente
        msg.y = y   # positivo = esquerda
        msg.w = w   # positivo = anti-horário
        msg.posType = pb.protoPositionSSL.dock

        self.udp_sock.sendto(msg.SerializeToString(), (self.device_address, self.device_port))


    def sendTargetPosition(self, x, y, w):
        msg = pb.protoPositionSSL()

        msg.x = x   # positivo = frente
        msg.y = y   # positivo = esquerda
        msg.w = w   # positivo = anti-horário
        msg.posType = pb.protoPositionSSL.target

        self.udp_sock.sendto(msg.SerializeToString(), (self.device_address, self.device_port))

    def sendSourcePosition(self, x, y, w):
        msg = pb.protoPositionSSL()

        msg.x = x   # positivo = frente
        msg.y = y   # positivo = esquerda
        msg.w = w   # positivo = anti-horário
        msg.posType = pb.protoPositionSSL.source

        self.udp_sock.sendto(msg.SerializeToString(), (self.device_address, self.device_port))


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

    x = 0
    y = 0
    w = 0

    print(f"Sending X, Y,  W Position")
    print(f"x = {x}, y = {y}, w = {w}")
    
    UDP.sendPosition(x, y, w)
