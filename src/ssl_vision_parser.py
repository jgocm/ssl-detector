import socket
import ssl_vision_wrapper_pb2

class FieldInformation:

# FieldInformation : {
#     cameraId: [
#         t_capture : float
#         balls : []
#         blueRobots : []
#         yellowRobots: []
#     ]
# }

    def __init__(self):
        self.cameras = {}

    def update(self, frame):
        self.resetCamera(frame.camera_id)
        for ball in frame.balls:
            self.updateBall(frame.camera_id, ball)

        for robot in frame.robots_yellow:
            self.updateRobot(frame.camera_id, robot, 'yellowRobots')

        for robot in frame.robots_blue:
            self.updateRobot(frame.camera_id, robot, 'blueRobots')

        self.cameras[frame.camera_id]['t_capture'] = frame.t_capture

    def resetCamera(self, cameraId):
        self.cameras[cameraId] = {}
        self.cameras[cameraId]['balls'] = []
        self.cameras[cameraId]['blueRobots'] = []
        self.cameras[cameraId]['yellowRobots'] = []
        self.cameras[cameraId]['t_capture'] = 0.0

    def updateBall(self, cameraId, ball):
        tempBall = {}
        tempBall['x'] = ball.x
        tempBall['y'] = ball.y
        self.cameras[cameraId]['balls'].append(tempBall)

    def updateRobot(self, cameraId, yellowRobot, teamColor):
        tempRobot = {}
        tempRobot['id'] = yellowRobot.robot_id
        tempRobot['x'] = yellowRobot.x
        tempRobot['y'] = yellowRobot.y
        tempRobot['orientation'] = yellowRobot.orientation
        self.cameras[cameraId][teamColor].append(tempRobot)

    def getAll(self, cameraId = 0):
        # field : {
        #     t_capture : float
        #     balls : []
        #     blueRobots : []
        #     yellowRobots: []
        # }

        # TODO: Como retornar o t_capture de varias cameras?
        # Acho que no caso dos mestrados, a gente so vai querer de uma camera, ai teria que dizer o id dela

        # TODO: Nao colocar nada na interseccao de camera, senao vai retornar repetido
        # Nao tem tratamento para isso

        field = {}
        field['balls'] = []
        field['blueRobots'] = []
        field['yellowRobots'] = []

        camera = self.cameras[cameraId]
        for ball in camera['balls']:
            field['balls'].append(ball)

        for robot in camera['blueRobots']:
            field['blueRobots'].append(robot)

        for robot in camera['yellowRobots']:
            field['yellowRobots'].append(robot)

        field['t_capture'] = camera['t_capture']

        field['blueRobots'] = sorted(field['blueRobots'], key=lambda d: d['id'])
        field['yellowRobots'] = sorted(field['yellowRobots'], key=lambda d: d['id'])

        return field

class SSLClient:
    
    def __init__(self, ip = '172.20.30.191', port=10006):
        """
        Init SSLClient object.

        Extended description of function.

        Parameters
        ----------
        ip : str
            Multicast IP in format '255.255.255.255'. 
        port : int
            Port up to 1024. 
        """
        
        self.ip = ip
        self.port = port

    def connect(self):
        """Binds the client with ip and port and configure to UDP multicast."""

        if not isinstance(self.ip, str):
            raise ValueError('IP type should be string type')
        if not isinstance(self.port, int):
            raise ValueError('Port type should be int type')
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 128)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        self.sock.bind((self.ip, self.port))

        host = socket.gethostbyname(socket.gethostname())
        self.sock.setsockopt(socket.SOL_IP, socket.IP_MULTICAST_IF, socket.inet_aton(host))
        self.sock.setsockopt(socket.SOL_IP, socket.IP_ADD_MEMBERSHIP, 
                socket.inet_aton(self.ip) + socket.inet_aton(host))

    def forceConnect(self, ip = '172.20.30.191', port = 10006):
        """Binds the client to its own ip and port."""

        if not isinstance(ip, str):
            raise ValueError('IP type should be string type')
        if not isinstance(port, int):
            raise ValueError('Port type should be int type')

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.bind((ip, port))

    def receive(self):
        """Receive package and decode."""
        # TODO: fix wrapper for geometry packets
        decoded_data = None
        has_package = False
        # while True:
        try:
            data, _ = self.sock.recvfrom(1024)
            decoded_data = ssl_vision_wrapper_pb2.SSL_WrapperPacket().FromString(data)
            has_package = True
        except:
            has_package = False 

        return has_package, decoded_data

def main():
    c = SSLClient(port=10006)
    c.forceConnect()
    print("connected")
    field = FieldInformation()

    while True:
        ret, pkg = c.receive()
        if ret:
            field.update(pkg.detection)
            f = field.getAll(pkg.detection.camera_id)
            for ball in f['balls']:
                if ball['y'] < 0:
                    print(f'camera: {pkg.detection.camera_id}, ball:{ball}')
            for robot_yellow in f['yellowRobots']:
                if robot_yellow['y'] < 0:
                    print(f'robot yellow {robot_yellow}')

if __name__ == '__main__':
    main()

