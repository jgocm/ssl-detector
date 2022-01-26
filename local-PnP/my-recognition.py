import jetson.inference
import jetson.utils

import argparse

# parse the command line
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, ect. (see --help for others)")
args = parser.parse_args()

# load image
img = jetson.utils.loadImage(args.filename)

# load network
net = jetson.inference.imageNet(args.network)

# classify the image
class_idx, confidence = net.Classify(img)

# get class name
class_name = net.GetClassDesc(class_idx)

# print result
print('classfied image as {:s} (class ID {:d}, confidence {:f})'.format(class_name, class_idx, confidence))

