import argparse
import os
import logging

import caffe
from eyewitness.dataset_util import BboxDataSet
from eyewitness.evaluation import BboxMAPEvaluator

from naive_detector import PeLeeDetectorWrapper


parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
'''
Command line options
'''
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--repo_path', type=str, default='/opt/caffe/examples/pelee')
parser.add_argument(
    '--db_path', type=str, default='::memory::',
    help='the path used to store detection result records'
)
parser.add_argument(
    '--interval_s', type=int, default=3, help='the interval of image generation'
)

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    args = parser.parse_args()

    # gpu preparation
    if args.gpu_id >= 0:
        caffe.set_device(args.gpu_id)
        caffe.set_mode_gpu()
    
    # setup your detector
    repo_path = args.repo_path
    params = {
        'labelmap_file': os.path.join(repo_path, 'model/voc/labelmap_voc.prototxt'),
        'model_def': os.path.join(repo_path, 'model/voc/deploy_merged.prototxt'),
        'model_weights': os.path.join(repo_path, 'model/voc/pelee_merged.caffemodel'),
    }
    object_detector = PeLeeDetectorWrapper(params, threshold=0.6)

    dataset_folder = 'VOC2007'
    dataset_VOC_2007 = BboxDataSet(dataset_folder, 'VOC2007')
    bbox_map_evaluator = BboxMAPEvaluator(test_set_only=False)
    # which will lead to ~0.58
    print(bbox_map_evaluator.evaluate(object_detector, dataset_VOC_2007)['mAP'])
