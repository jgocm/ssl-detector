import tensorrt as trt
import numpy as np
import time
import random
import trt_common
import colorsys
import cv2
import os

class Model():
    def __init__(
                self,
                context,
                inputs,
                outputs,
                bindings,
                stream
                ):
        super(Model, self).__init__()
        self.context = context
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.stream = stream


class DetectNet():
    def __init__(
                self, 
                model_path="../models/ssdlite_mobilenet_v2_300x300_ssl_fp16.trt", 
                labels_path="../models/ssl_labels.txt", 
                input_width=300, 
                input_height=300,
                score_threshold = 0.5,
                draw = False,
                display_fps = True,
                TRT_LOGGER = trt.Logger(trt.Logger.INFO)
                ):
        super(DetectNet,self).__init__()
        self.model_path = model_path
        self.labels_path = labels_path
        self.input_width = input_width
        self.input_height = input_height
        self.score_threshold = score_threshold
        self.draw = draw
        self.display_fps = display_fps

        self.TRT_LOGGER = TRT_LOGGER
        trt.init_libnvinfer_plugins(self.TRT_LOGGER, "")

        self.model = Model(
                context=None,
                inputs=None,
                outputs=None,
                bindings=None,
                stream=None
        )
        self.labels = None
        self.colors = None
        self.elapsed_list = []
        self.detections = []
    
    def readLabelFile(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
        ret = {}
        for line in lines:
            pair = line.strip().split(maxsplit=1)
            ret[int(pair[0])] = pair[1].strip()
        return ret

    def randomColors(self, N):
        N = N + 1
        hsv = [(i / N, 1.0, 1.0) for i in range(N)]
        colors = list(
            map(lambda c: tuple(int(i * 255) for i in colorsys.hsv_to_rgb(*c)), hsv)
        )
        random.shuffle(colors)
        return colors

    def readLabels(self):
        labels = self.readLabelFile(self.labels_path)
        self.labels = labels
        last_key = sorted(labels.keys())[len(labels.keys()) - 1]
        random.seed(42)
        colors = self.randomColors(last_key)
        self.colors = colors

    def getEngine(self, engine_file_path):
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def loadModel(self):
        self.readLabels()
        engine = self.getEngine(self.model_path)
        context = engine.create_execution_context()
        inputs, outputs, bindings, stream = trt_common.allocate_buffers(engine)
        self.model = Model(
                          context = context,
                          inputs = inputs,
                          outputs = outputs,
                          bindings = bindings,
                          stream = stream
                          )
        
        return self
    
    def draw_rectangle(self, image, box, color, thickness=2):
        b = np.array(box).astype(int)
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness)

    def draw_caption(self, image, box, caption):
        b = np.array(box).astype(int)
        cv2.putText(
            image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )
        cv2.putText(
            image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )

    def normalize(self, img):
        img = np.asarray(img, dtype="float32")
        img = img / 127.5 - 1.0
        return img
        
    '''    
    def convertImage(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_im = cv2.resize(img, (self.input_width, self.input_height))
        normalized_im = self.normalize(resized_im)
        normalized_im = np.expand_dims(normalized_im, axis=0)
        return normalized_im
    '''

    def convertImage(self, img, width, height):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_im = cv2.resize(img, (width, height))
        normalized_im = self.normalize(resized_im)
        normalized_im = np.expand_dims(normalized_im, axis=0)
        return normalized_im

    def inference(self, img):
        # ORIGINAL IMAGE DIMENSIONS
        capture_height = int(img.shape[0])
        capture_width = int(img.shape[1]) 

        # CONVERT IMAGE FORMAT
        cvt_img = self.convertImage(img, 
                                width=self.input_width,
                                height=self.input_height
                                )

        start = time.perf_counter()
        self.model.inputs[0].host = cvt_img
        trt_outputs = trt_common.do_inference_v2(
            context=self.model.context, 
            bindings=self.model.bindings, 
            inputs=self.model.inputs, 
            outputs=self.model.outputs, 
            stream=self.model.stream
        )
        inference_time = (time.perf_counter() - start) * 1000

        boxes = trt_outputs[1].reshape([-1, 4])
        detections = []
        for index in range(int(trt_outputs[0])):
            box = boxes[index]
        for index, box in enumerate(boxes):
            if trt_outputs[2][index] < self.score_threshold:
                continue
        
            # BOUNDING BOX COORDINATES
            class_id = int(trt_outputs[3][index])
            score = trt_outputs[2][index]

            xmin = int(box[0] * capture_width)
            xmax = int(box[2] * capture_width)
            ymin = int(box[1] * capture_height)
            ymax = int(box[3] * capture_height)
            detection = (class_id, score, xmin, xmax, ymin, ymax)
            detections.append(detection)
            if self.draw:
                caption = "{0}({1:.2f})".format(self.labels[class_id - 1], score)
                self.draw_rectangle(img, (xmin, ymin, xmax, ymax), self.colors[class_id - 1])
                #self.draw_caption(img, (xmin, ymax - 5), caption)
        self.detections = detections

        # COMPUTE AVG FPS
        '''self.elapsed_list.append(inference_time)
        avg_text = ""
        if len(self.elapsed_list) > 100:
            self.elapsed_list.pop(0)
            avg_elapsed_ms = np.mean(self.elapsed_list)
            avg_text = " AGV: {0:.2f}ms".format(avg_elapsed_ms)
        
        # DISPLAY FPS
        fps_text = "Inference: {0:.2f}ms".format(inference_time)
        display_text = fps_text + avg_text
        if self.display_fps: self.draw_caption(img, (10, 30), display_text)'''
        return self

if __name__ == "__main__":

    DISPLAY_WINDOW = True
    WINDOW_NAME = 'Object Detection'
    cap = cv2.VideoCapture("/dev/video0")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cwd = os.getcwd()

    trt_net = DetectNet(
                model_path=cwd+"/models/ssdlite_mobilenet_v2_300x300_ssl_fp16.trt", 
                labels_path=cwd+"/models/ssl_labels.txt", 
                input_width=300, 
                input_height=300,
                score_threshold = 0.5,
                draw = True,
                display_fps = True,
                TRT_LOGGER = trt.Logger(trt.Logger.INFO)
                )
    
    trt_net.loadModel()

    start_time = time.time()

    while True:
        if cap.isOpened():
           ret, frame = cap.read()
           if not ret:
               print("Check video capture path")
               break
           else: img = frame

        detections = trt_net.inference(img).detections

        for detection in detections:
            class_id, score, xmin, xmax, ymin, ymax = detection
            print(f"Class ID: {class_id} | Score: {score} | Bounding Box: {xmin}, {ymin}, {xmax}, {ymax}")

        # DISPLAY WINDOW
        if DISPLAY_WINDOW:
            cv2.moveWindow(WINDOW_NAME, 100, 50)
            cv2.imshow(WINDOW_NAME, img)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
            
        elif time.time() - start_time > 10:
            break

    # RELEASE WINDOW AND DESTROY
    cap.release()
    cv2.destroyAllWindows()


