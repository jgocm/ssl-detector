#!/usr/bin/python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
from object_detection import detectnet

import jetson.inference
import jetson.utils

import argparse
import sys
import os

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="/dev/video0", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# create video output object 
output = jetson.utils.videoOutput("display://0")
	
# load the object detection network
net_args = sys.argv
#net_args.extend([ssl_mb2_model,
#				ssl_labels,
#				batch_size,
#				input_blob,
#				output_cvg,
#				output_bbox
#				])
#net = jetson.inference.detectNet(opt.network, net_args, opt.threshold)
#net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# network args
#path=os.path.dirname(os.path.abspath(__file__)) #parent folder path
path = os.path.dirname(os.getcwd())
ssl_mb2_model='/home/joao/ssl-detector/models/ssl3/mb2-ssd-lite.onnx'
ssl_labels='/home/joao/ssl-detector/models/labels.txt'
batch_size ='--batch-size=4'
input_blob='--input-blob=input_0'
output_cvg='--output-cvg=scores'
output_bbox='--output-bbox=boxes'

net = detectnet(network="ssd-mobilenet-v2",
				path_to_model=ssl_mb2_model,
				path_to_labels=ssl_labels,
				batch_size=4,
				).net

# create video sources
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)


# process frames until the user exits
while True:
	# capture the next image
	img = input.Capture()

	# detect objects in the image (with overlay)
	detections = net.Detect(img, overlay=opt.overlay)

	# print the detections
	#print("detected {:d} objects in image".format(len(detections)))

	for detection in detections:
		print(detection)
		ID = detection.ClassID
		top = int(detection.Top)
		bottom = int(detection.Bottom)
		left = int(detection.Left)
		right = int(detection.Right)
		#jetson.utils.cudaDrawRect(img, (left, top, right, bottom), (0,255,0,255))

	# render the image
	output.Render(img)
	
	# update the title bar
	output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))
	
	# print out performance info
	#net.PrintProfilerTimes()

	# exit on input/output EOS
	if not input.IsStreaming() or not output.IsStreaming():
		break

