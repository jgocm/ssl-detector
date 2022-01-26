from cv2 import threshold
import jetson.inference
import jetson.utils

import argparse
import sys
import os

class detectnet():
    def __init__(
                self, 
                network="ssd-mobilenet-v2", 
                path_to_model=None, 
                path_to_labels=None, 
                batch_size=4,
                input_blob = 'input_0',
                output_cvg = 'scores',
                output_bbox = 'boxes',
                threshold = 0.5
                ):
        super(detectnet,self).__init__()
        self.net_args = sys.argv
        self.net_args.extend(["--model="+path_to_model,
				"--labels="+path_to_labels,
				"--batch-size="+str(batch_size),
				"--input-blob="+input_blob,
				"--output-cvg="+output_cvg,
				"--output-bbox="+output_bbox
				])
        self.net = jetson.inference.detectNet(network, self.net_args, threshold)

