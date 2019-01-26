import argparse
import os
import sys
import numpy as np

import caffe
import arrow
import PIL
from caffe.proto import caffe_pb2
from eyewitness.detection_utils import DetectionResult
from eyewitness.image_id import ImageId
from eyewitness.object_detector import ObjectDetector
from eyewitness.image_utils import (ImageHandler, Image)
from google.protobuf import text_format


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
        self.transformer.set_mean('data', np.array([103.94, 116.78, 123.68]))  # mean pixel
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
            for i in range(0, num_labels):
                if label == labelmap.item[i].label:
                    found = True
                    labelnames.append(labelmap.item[i].display_name)
                    break
            assert found
        return labelnames

    def detect(self, image_obj):
        """
        need to implement detection method which return DetectionResult obj

        Parameters
        ----------
        image_obj: eyewitness.image_utils.Image
            eyewitness image obj

        Returns
        -------
        DetectionResult
        """
        results = self.predict(np.array(image_obj.pil_image_obj))

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
            'image_id': image_obj.image_id,
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
    parser.add_argument('--repo_path', type=str, default='/opt/caffe/examples/pelee')
    args = parser.parse_args()
    repo_path = args.repo_path
    # gpu preparation
    if args.gpu_id >= 0:
        caffe.set_device(args.gpu_id)
        caffe.set_mode_gpu()

    params = {
        'labelmap_file': os.path.join(repo_path, 'model/voc/labelmap_voc.prototxt'),
        'model_def': os.path.join(repo_path, 'model/voc/deploy_merged.prototxt'),
        'model_weights': os.path.join(repo_path, 'model/voc/pelee_merged.caffemodel'),
    }

    object_detector = PeLeeDetectorWrapper(params, threshold=0.6)

    raw_image_path = os.path.join(repo_path, 'samples/5566.jpg')
    image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
    image_obj = Image(image_id, raw_image_path=raw_image_path)
    detection_result = object_detector.detect(image_obj)
    ImageHandler.draw_bbox(image_obj.pil_image_obj, detection_result.detected_objects)
    ImageHandler.save(image_obj.pil_image_obj, "detected_image/drawn_image.jpg")
