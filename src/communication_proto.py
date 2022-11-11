import socket
import CommTypes_pb2 as pb
from navigation import TargetPoint
from entities import Robot

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
        self.udp_sock.bind(('', self.host_port))
        self.udp_sock.settimeout(0)
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

    def sendTargetPosition(self, x, y, w):
        msg = self.setKickMessage()
        msg.x = x
        msg.y = y
        msg.w = w
        msg.posType = pb.protoPositionSSL.driveToTarget

        self.udp_sock.sendto(msg.SerializeToString(), (self.device_address, self.device_port))

    def sendSourcePosition(self, x, y, w):
        msg = self.setKickMessage()
        msg.x = x   # positivo = frente
        msg.y = y   # positivo = esquerda
        msg.w = w   # positivo = anti-horário
        msg.posType = pb.protoPositionSSL.source

        self.udp_sock.sendto(msg.SerializeToString(), (self.device_address, self.device_port))

    def resetRobotPosition(self):
        msg = pb.protoPositionSSL()
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
     
    def setPositionMessage(self, x, y, w, pos_type, reset_odometry):
        self.msg.x = x
        self.msg.y = y
        self.msg.w = w
        self.msg.posType = pos_type
        self.msg.resetOdometry = reset_odometry

        return self.msg

    def setSSLMessage(self, target=TargetPoint(), robot=Robot()):
        self.msg = pb.protoPositionSSL()
        self.msg.x = target.x
        self.msg.y = target.y
        self.msg.w = target.w
        self.msg.max_speed = target.max_speed
        self.msg.min_speed = target.min_speed
        self.msg.posType = target.type
        self.msg.resetOdometry = target.reset_odometry

        self.msg.front = robot.front
        self.msg.chip = robot.chip
        self.msg.charge = robot.charge
        self.msg.kickStrength = robot.kick_strength
        self.msg.dribbler = robot.dribbler
        self.msg.dribSpeed = robot.dribbler_speed

        return self.msg

    def sendSSLMessage(self, times = 3):
        for i in range(0, times):
            self.udp_sock.sendto(self.msg.SerializeToString(), (self.device_address, self.device_port))
    
    def recvSSLMessage(self):
        msg = pb.protoOdometry()
        # multiple messages are received and accumulated on buffer during vision processing
        # so read until buffer socket are no longer available
        while True:
            try:
                data, _ = self.udp_sock.recvfrom(1024)
                msg.ParseFromString(data)
            except:
                break 
        movement = [msg.x, msg.y, msg.w]
        
        return movement, msg.hasBall, msg.kickLoad, msg.battery, msg.count


if __name__ == "__main__":
    import time, math

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

    SEND = False

    if SEND:
        x = 1
        y = 0
        w = 0

        print(f"Sending X, Y, W Position")
        print(f"x = {x}, y = {y}, w = {w}")

        UDP.msg.posType = pb.protoPositionSSL.driveToTarget
        UDP.msg.x = x
        UDP.msg.y = y
        UDP.msg.w = w
        UDP.msg.max_speed = 1

        while(1):
            UDP.sendSSLMessage()
            time.sleep(0.033)
    
    else:
        while True:
            odometry, hasBall, kickLoad, battery, count = UDP.recvSSLMessage()
            if battery>15:
                print(f"{odometry}, {hasBall}, {kickLoad:.3f}, {battery:.3f}, {count}")