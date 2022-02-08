import socket
import math

def decodePacket(data, pos_size=20):
    mask = 2**(pos_size-1)-1

    ball_detect = data & 1
    data = data >> 1
    robot_detect = data & 1
    data = data >> 1
    goal_detect = data & 1
    data = data >> 1
    detected_objs = [ball_detect, robot_detect, goal_detect]

    signed = (data & (mask+1))>>(pos_size-1)
    ballX = data & mask
    if signed: ballX = -ballX
    data = data >> pos_size

    signed = data & (mask+1)
    ballY = data & mask
    if signed: ballY = -ballY
    data = data >> pos_size
    ball_position = [ballX, ballY]

    signed = data & (mask+1)
    robotX = data & mask
    if signed: robotX = -robotX
    data = data >> pos_size

    signed = data & (mask+1)
    robotY = data & mask
    if signed: robotY = -robotY
    data = data >> pos_size
    robot_position = [robotX, robotY]

    signed = data & (mask+1)
    goalX = data & mask
    if signed: goalX = -goalX
    data = data >> pos_size

    signed = data & (mask+1)
    goalY = data & mask
    if signed: goalY = -goalY
    data = data >> pos_size
    goal_position = [goalX, goalY]

    return detected_objs,ball_position,robot_position,goal_position
    
def int_to_bytes(i: int, *, signed: bool = False) -> bytes:
    length = ((i + ((i * signed) < 0)).bit_length() + 7 + signed) // 8
    return i.to_bytes(length, byteorder='big', signed=signed)

def bytes_to_int(b: bytes, *, signed: bool = False) -> int:
    return int.from_bytes(b, byteorder='big', signed=signed)

# Test unsigned:
for i in range(1025):
    assert i == bytes_to_int(int_to_bytes(i))

# Test signed:
for i in range(-1024, 1025):
    assert i == bytes_to_int(int_to_bytes(i, signed=True), signed=True)

UDP_IP = "172.20.10.2"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))
pos_size=20

while True:
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    data_int = bytes_to_int(data)
    detected_objs,ball_position,robot_position,goal_position = decodePacket(data_int, pos_size)
    print(detected_objs,ball_position,robot_position,goal_position)

    if detected_objs[1]: print(f'Ball Position: {ball_position}')
    if detected_objs[2]: print(f'Robot Position: {robot_position}')
    if detected_objs[0]: print(f'Goal center position: {goal_position}')