import socket
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
        self.msg = pb.protoPositionSSL()

    def setHostUDP(self, address, port):
        self.host_address=address
        self.host_port=port
    
    def setDeviceUDP(self, address,port):
        self.device_address=address
        self.device_port=port

    def sendStopMotion(self):
        msg = self.setKickMessage()
        msg.x = 0
        msg.y = 0
        msg.w = 0
        msg.posType = pb.protoPositionSSL.stop

        self.udp_sock.sendto(msg.SerializeToString(), (self.device_address, self.device_port))

    def sendRotateSearch(self):
        msg = self.setKickMessage()
        msg.x = 0
        msg.y = 0
        msg.w = 0
        msg.posType = pb.protoPositionSSL.rotateOnSelf

        self.udp_sock.sendto(msg.SerializeToString(), (self.device_address, self.device_port))

    def sendRotateInPoint(self, x, y, w):
        self.msg.x = x   # positivo = frente
        self.msg.y = y   # positivo = esquerda
        self.msg.w = w   # positivo = anti-horário
        self.msg.posType = pb.protoPositionSSL.rotateInPoint

        self.udp_sock.sendto(self.msg.SerializeToString(), (self.device_address, self.device_port))

    def sendRotateControl(self, x, y, w):
        msg = self.setKickMessage()
        msg.x = x
        msg.y = y
        msg.w = w
        msg.posType = pb.protoPositionSSL.rotateControl

        self.udp_sock.sendto(msg.SerializeToString(), (self.device_address, self.device_port))

    def sendBallDocking(self, x, y, w):
        msg = self.setKickMessage()
        msg.x = x
        msg.y = y
        msg.w = w
        msg.posType = pb.protoPositionSSL.dock

        self.udp_sock.sendto(msg.SerializeToString(), (self.device_address, self.device_port))

    def sendTargetPosition(self, x, y, w):
        msg = self.setKickMessage()
        msg.x = x
        msg.y = y
        msg.w = w
        msg.posType = pb.protoPositionSSL.target

        self.udp_sock.sendto(msg.SerializeToString(), (self.device_address, self.device_port))

    def sendSourcePosition(self, x, y, w):
        msg = self.setKickMessage()
        msg.x = x   # positivo = frente
        msg.y = y   # positivo = esquerda
        msg.w = w   # positivo = anti-horário
        msg.posType = pb.protoPositionSSL.source

        self.udp_sock.sendto(msg.SerializeToString(), (self.device_address, self.device_port))

    def resetRobotPosition(self):
        msg = self.setKickMessage()
        msg.x = 0
        msg.y = 0
        msg.w = 0
        msg.posType = pb.protoPositionSSL.source

        self.udp_sock.sendto(msg.SerializeToString(), (self.device_address, self.device_port))

    def setKickMessage(self, front=False, chip=False, charge=False, kickStrength=0, dribbler=False, dribSpeed=0):
        self.msg.front = front
        self.msg.chip = chip
        self.msg.charge = charge
        self.msg.kickStrength = kickStrength
        self.msg.dribbler = dribbler
        self.msg.dribSpeed = dribSpeed

        return self.msg
     
    def setPositionMessage(self, x, y, w, posType):
        self.msg.x = x
        self.msg.y = y
        self.msg.w = w
        self.msg.posType = posType

        return self.msg

    def sendSSLMessage(self):
        self.udp_sock.sendto(self.msg.SerializeToString(), (self.device_address, self.device_port))

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
