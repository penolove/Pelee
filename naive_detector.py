import argparse
import os
import sys
import numpy as np

import caffe
import arrow
from caffe.proto import caffe_pb2
from eyewitness.detection_utils import DetectionResult
from eyewitness.image_id import ImageId
from eyewitness.object_detector import ObjectDetector
from eyewitness.image_utils import (ImageHandler, swap_channel_rgb_bgr)
from google.protobuf import text_format
from PIL import Image


class PeLeeDetectorWrapper(ObjectDetector):
    def __init__(self, params, threshold=0.6):
        self.threshold = threshold

        self.labelmap = caffe_pb2.LabelMap()
        labelmap_file = params['labelmap_file']
        with open(labelmap_file, 'r') as f:
            text_format.Merge(str(f.read()), self.labelmap)

        # load model
        model_def = params['model_def']
        model_weights = params['model_weights']

        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)

        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_input_scale('data', 0.017)
        self.transformer.set_mean('data', np.array([103.94,116.78,123.68])) # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', 255)
        # the reference model has channels in BGR order instead of RGB
        self.transformer.set_channel_swap('data', (2, 1, 0))

    def predict(self, image_array):
        transformed_image = self.transformer.preprocess('data', image_array)
        self.net.blobs['data'].data[...] = transformed_image

        detections = self.net.forward()['detection_out']
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3] * image_array.shape[1]
        det_ymin = detections[0, 0, :, 4] * image_array.shape[0]
        det_xmax = detections[0, 0, :, 5] * image_array.shape[1]
        det_ymax = detections[0, 0, :, 6] * image_array.shape[0]

        result = np.column_stack([det_xmin, det_ymin, det_xmax, det_ymax, det_conf, det_label])
        return result

    def get_labelname(self, labelmap, labels):
        num_labels = len(labelmap.item)
        labelnames = []
        if type(labels) is not list:
            labels = [labels]
        for label in labels:
            found = False
            for i in xrange(0, num_labels):
                if label == labelmap.item[i].label:
                    found = True
                    labelnames.append(labelmap.item[i].display_name)
                    break
            assert found
        return labelnames

    def detect(self, image, image_id):
        """
        need to implement detection method which return DetectionResult obj

        Parameters
        ----------
        image: PIL.Image
            PIL.Image instance
        image_id: Union[str, ImageId]
            image_id

        Returns
        -------
        DetectionResult
        """
        results = self.predict(image_array)

        detected_objects = []
        for i in range(0, results.shape[0]):
            score = results[i, -2]
            if score < self.threshold:
                continue

            label_index = int(results[i, -1])
            label = self.get_labelname(self.labelmap, label_index)[0]

            x1 = int(round(results[i, 0]))
            y1 = int(round(results[i, 1]))
            x2 = int(round(results[i, 2]))
            y2 = int(round(results[i, 3]))

            detected_objects.append([x1, y1, x2, y2, label, score, ''])

        image_dict = {
            'image_id': image_id,
            'detected_objects': detected_objects,
        }
        detection_result = DetectionResult(image_dict)

        return detection_result


if __name__ == '__main__':
    # Make sure that caffe is on the python path:
    caffe_root = './'
    os.chdir(caffe_root)
    sys.path.insert(0, os.path.join(caffe_root, 'python'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    # gpu preparation
    if args.gpu_id >= 0:
        caffe.set_device(args.gpu_id)
        caffe.set_mode_gpu()

    params = {
        'labelmap_file': 'model/voc/labelmap_voc.prototxt',
        'model_def': 'model/voc/deploy_merged.prototxt',
        'model_weights': 'model/voc/pelee_merged.caffemodel',
    }

    object_detector = PeLeeDetectorWrapper(params, threshold=0.6)

    image = Image.open('samples/5566.jpg')
    image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
    detection_result = object_detector.detect(image, image_id)
    ImageHandler.draw_bbox(image, detection_result.detected_objects)
    ImageHandler.save(image, "detected_image/drawn_image.jpg")
