from socket import socket, AF_INET, SOCK_DGRAM
import CommTypes_pb2 as pb

scktPort = 9601

scktReceiver = socket(AF_INET, SOCK_DGRAM)
scktReceiver.bind(('',scktPort))

feedbackMsg = pb.protoOdometryFeedback()

while True:
    print("oi")
    message, clientAddress = scktReceiver.recvfrom(1024)
    print("oi 2")
    feedbackMsg.ParseFromString(message)    
    print("oi 3")
    print("x = {0}".format(feedbackMsg.x))
    print("y = {0}".format(feedbackMsg.y))
    print("w = {0}".format(feedbackMsg.w))/8

    