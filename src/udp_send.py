import socket

def shapeData(ball_position,robot_position,goal_position):

    ball_x, ball_y, ball_z = ball_position
    ball = [int(ball_x),int(ball_y)]

    robot_x, robot_y, robot_z = robot_position
    robot = [int(robot_x),int(robot_y)]

    goal_x, goal_y, goal_z = goal_position   
    goal = [int(goal_x),int(goal_y)]

    return ball, robot, goal

def encodePacket(detections, ball_position, robot_position, goal_position, pos_size=20):
    mask = 2**(pos_size-1)-1

    data_length = 0
    ball_detect = detections[0]
    data_length += 1
    robot_detect = detections[1]<<data_length
    data_length += 1
    goal_detect = detections[2]<<data_length
    data_length += 1
    detections_bin = ball_detect + robot_detect + goal_detect

    ball_x, ball_y = ball_position
    if ball_x<0: 
        signed=1
        ball_x=-ball_x
    else: signed=0
    ball_x = (ball_x | (signed<<pos_size-1))<<data_length
    data_length += pos_size

    if ball_y<0: 
        signed=1
        ball_y=-ball_y
    else: signed=0
    ball_y = (ball_y | (signed<<pos_size-1))<<data_length
    data_length += pos_size

    robot_x, robot_y = robot_position
    if robot_x<0: 
        signed=1
        robot_x=-robot_x
    else: signed=0
    robot_x = (robot_x | (signed<<pos_size-1))<<data_length
    data_length += pos_size

    if robot_y<0: 
        signed=1
        robot_y=-robot_y
    else: signed=0
    robot_y = (robot_y | (signed<<pos_size-1))<<data_length
    data_length += pos_size

    goal_x, goal_y = goal_position
    if goal_x<0: 
        signed=1
        goal_x=-goal_x
    else: signed=0
    goal_x = (goal_x | (signed<<pos_size-1))<<data_length
    data_length += pos_size

    if goal_y<0: 
        signed=1
        goal_y=-goal_y
    else: signed=0
    goal_y = (goal_y | (signed<<pos_size-1))<<data_length
    data_length += pos_size

    data = detections_bin+ball_x+ball_y+robot_x+robot_y+goal_x+goal_y
    return data

def int_to_bytes(i: int, *, signed: bool = False) -> bytes:
    length = ((i+(signed*i<0)).bit_length() + 7 + signed) // 8
    return i.to_bytes(length, byteorder='big', signed=signed)

def bytes_to_int(b: bytes, *, signed: bool = False) -> int:
    return int.from_bytes(b, byteorder='big', signed=signed)

# Test unsigned:
for i in range(1025):
    assert i == bytes_to_int(int_to_bytes(i))

# Test signed:
for i in range(-1024, 1025):
    assert i == bytes_to_int(int_to_bytes(i, signed=True), signed=True)


if __name__=='__main__':
    UDP_IP ='172.20.10.2'
    #UDP_IP ='192.168.1.7'
    UDP_PORT = 5005

    sock = socket.socket(socket.AF_INET,
                        socket.SOCK_DGRAM)

    pos_size = 20

    detections = [1,1,0]
    ball_position = [-8,4]
    robot_position = [4,2]
    goal_position = [2,1]

    data = encodePacket(detections, ball_position, robot_position, goal_position, pos_size=pos_size)
    print(data)

    bytes_data = int_to_bytes(data)

    while True:
        sock.sendto(bytes_data, (UDP_IP, UDP_PORT))
        print(data)