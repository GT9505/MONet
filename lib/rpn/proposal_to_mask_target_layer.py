import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
DEBUG = False
import cv2

class Proposal2MaskTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self._num_classes = layer_params['num_classes']

        # sampled positive rois for mask branch (0, x1, y1, x2, y2)
        top[0].reshape(1, 5, 1, 1)
        # labels for mask branch
        top[1].reshape(1, self._num_classes, cfg.TRAIN.MASK_RESOLUTION, cfg.TRAIN.MASK_RESOLUTION)

    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_target_layer.ProposalTargetLayer), or any other source
        proposals = bottom[0].data
        proposal_labels = bottom[1].data
        assert(proposals.shape[0] == proposal_labels.shape[0])
        gt_boxes = bottom[2].data
        gt_masks = bottom[3].data
        
        index = np.where(proposal_labels>0)[0]
        fg_rois_for_mask_per_this_image = min(cfg.TRAIN.FG_FOR_MASK,index.shape[0])
        index = npr.choice(index, size=fg_rois_for_mask_per_this_image, replace=False)
        
        proposals = proposals[index,:]
        proposal_labels = proposal_labels[index]
        
        overlaps = bbox_overlaps(
            np.ascontiguousarray(proposals[:, 1:5,0,0], dtype=np.float),
            np.ascontiguousarray(gt_boxes[:, :4,0,0], dtype=np.float))
        gt_assignment = overlaps.argmax(axis=1)
        assert((proposal_labels[:,0,0,0] == gt_boxes[gt_assignment,4,0,0]).all())
        
        proposal_mask_labels = np.zeros( \
            (proposals.shape[0], self._num_classes, cfg.TRAIN.MASK_RESOLUTION, cfg.TRAIN.MASK_RESOLUTION) \
        ) - 1
        mask_visual = []
        for i in range(proposals.shape[0]):
            mask = gt_masks[gt_assignment[i],:,:]
            mask = mask[ \
                int(np.round(proposals[i,2,0,0])):int(np.round(proposals[i,4,0,0])), \
                int(np.round(proposals[i,1,0,0])):int(np.round(proposals[i,3,0,0])) \
            ]
            mask_visual.append(mask)
            mask = cv2.resize(mask, dsize=(cfg.TRAIN.MASK_RESOLUTION, cfg.TRAIN.MASK_RESOLUTION), interpolation=cv2.INTER_LINEAR)
            proposal_mask_labels[i,int(proposal_labels[i]),:,:] = np.round(mask)

        # sampled positive rois for mask branch
        proposals = proposals.reshape((proposals.shape[0], proposals.shape[1], 1, 1))
        top[0].reshape(*proposals.shape)
        top[0].data[...] = proposals
        
        # labels for mask branch
        proposal_mask_labels = proposal_mask_labels.reshape((proposal_mask_labels.shape[0], self._num_classes, cfg.TRAIN.MASK_RESOLUTION, cfg.TRAIN.MASK_RESOLUTION))
        top[1].reshape(*proposal_mask_labels.shape)
        top[1].data[...] = proposal_mask_labels

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass