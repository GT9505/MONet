import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from utils.cython_bbox import bbox_overlaps
DEBUG = False
import cv2

class Proposal2PredictedBoxesLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        # predicted boxes
        top[0].reshape(1, 5)

    def forward(self, bottom, top):
        proposals = bottom[0].data
        proposals = proposals[:,1:]
        predicted_box_deltas = bottom[1].data
        predicted_box_deltas = predicted_box_deltas.reshape((-1,8))
        im_info = bottom[2].data
        
        pred_boxes = bbox_transform_inv(proposals, predicted_box_deltas[:,4:])
        pred_boxes = clip_boxes(pred_boxes, [im_info[0,0],im_info[0,1]])
        
        pred_boxes_final = np.zeros((pred_boxes.shape[0],5))
        pred_boxes_final[:,1:] = pred_boxes
        
        top[0].reshape(*pred_boxes_final.shape)
        top[0].data[...] = pred_boxes_final

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass