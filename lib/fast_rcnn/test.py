# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms, soft_nms
import cPickle
from utils.blob import im_list_to_blob
import os
import pdb
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools import mask as maskUtils

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_orig /= cfg.PIXEL_STDS

    im_shape = im_orig.shape
    h = im_orig.shape[0]
    w = im_orig.shape[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
            
        if cfg.TEST.IMAGE_BASE_SIZE > 1:
            resize_height = np.round((h * im_scale)/cfg.TRAIN.IMAGE_BASE_SIZE)*cfg.TRAIN.IMAGE_BASE_SIZE
            resize_width  = np.round((w * im_scale)/cfg.TRAIN.IMAGE_BASE_SIZE)*cfg.TRAIN.IMAGE_BASE_SIZE
            im = cv2.resize(im_orig, dsize=(int(resize_width), int(resize_height)),
                        interpolation=cv2.INTER_LINEAR)
            im_scale = np.array([resize_height/float(h),resize_width/float(w)])
        else:
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def im_detect(net, im, boxes=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        if cfg.TEST.IMAGE_BASE_SIZE > 1:
            blobs['im_info'] = np.array(
                [[im_blob.shape[2], im_blob.shape[3], im_scales[0][0],im_scales[0][1]]],
                dtype=np.float32)
        else:
            blobs['im_info'] = np.array(
                [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
                dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        
        if cfg.TEST.IMAGE_BASE_SIZE > 1:
            boxes = np.zeros((rois.shape[0],4))
            boxes[:,0:4:2] = rois[:, 1:5:2] / im_scales[0][1]
            boxes[:,1:4:2] = rois[:, 2:5:2] / im_scales[0][0]
        else:
            boxes = rois[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']  
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]
    
    if cfg.TEST.MASK_ON:
        masks = blobs_out['conv_mask_pred_prob']
        return scores, pred_boxes, masks
    else:
        return scores, pred_boxes

def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def bbox_vote(det):
    if det.shape[0] <= 1:
        return det
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    # det = det[np.where(det[:, 4] > 0.2)[0], :]
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these  det
        merge_index = np.where(o >= 0.45)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score
            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    return dets

def test_net(net, imdb, max_per_image=100, thresh=-np.inf, vis=False):
    """Test a Fast R-CNN network on an image database."""
    
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
                 
    if cfg.TEST.MASK_ON:
        all_masks = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer(),'im_detect1' : Timer(),'im_detect2' : Timer()}

    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb
  

    for i in xrange(num_images):
    #for i in xrange(1):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]     
        
        
        im = cv2.imread(imdb.image_path_at(i))
		
        scores = np.empty(shape=[0,cfg.DATASET_CATEGORY_NUMBER])
        if cfg.TEST.MASK_ON:
            masks = np.empty( shape=[0,cfg.DATASET_CATEGORY_NUMBER,cfg.TEST.MASK_RESOLUTION,cfg.TEST.MASK_RESOLUTION] )
        
        _t['im_detect'].tic()		
        for test_scale in np.array([600]):
            
            if cfg.TEST.MASK_ON:
                scores, boxes,masks = im_detect(net, im, box_proposals) 

            else:
                scores, boxes = im_detect(net, im, box_proposals) 
        _t['im_detect'].toc()

        _t['misc'].tic()
        # skip j = 0, because it's the background class
        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            
            if cfg.TEST.MASK_ON:
                cls_masks = masks[inds,j,:,:]
            
            if cfg.TEST.AGNOSTIC:
                cls_boxes = boxes[inds, 4:8]
            else:
                cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
                       
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            
            if cfg.TEST.MASK_ON:
                cls_masks = cls_masks[keep,:,:]
            
            if vis:
                vis_detections(im, imdb.classes[j], cls_dets)
            all_boxes[j][i] = cls_dets
            
            if cfg.TEST.MASK_ON:
                all_masks[j][i] = cls_masks

        # Limit to max_per_image detections *over all classes*
        
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
                    
                    if cfg.TEST.MASK_ON:
                        all_masks[j][i] = all_masks[j][i][keep,:,:]
             

        if cfg.TEST.MASK_ON:
            for j in xrange(1, imdb.num_classes):
                cls_masks = all_masks[j][i]
                cls_dets = all_boxes[j][i]
                cls_masks_encode = []
                
                for k in range(cls_dets.shape[0]):
                    per_box_height = int(cls_dets[k,3]) - int(cls_dets[k,1]) 
                    per_box_width  = int(cls_dets[k,2]) - int(cls_dets[k,0])
              
                    per_box_masks = cls_masks[k,:,:]
                    per_box_masks = cv2.resize(per_box_masks, dsize=(per_box_width, per_box_height),interpolation=cv2.INTER_LINEAR)
                    
                    per_box_masks_in_image = np.zeros((im.shape[0],im.shape[1]))
                    per_box_masks_in_image[int(cls_dets[k,1]):int(cls_dets[k,3]),int(cls_dets[k,0]):int(cls_dets[k,2])] = np.round(per_box_masks)
                    
                    rle = maskUtils.encode(np.array(per_box_masks_in_image[:, :, np.newaxis], order="F").astype(np.uint8))[0]
                    rle["counts"] = rle["counts"].decode("utf-8")

                    cls_masks_encode.append(rle)
                    cls_masks_encode.append(cls_dets[k,4])
                all_masks[j][i] = cls_masks_encode

        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)
         
    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)
    
    if cfg.TEST.MASK_ON:
        mask_file = os.path.join(output_dir, 'segmentations.pkl')
        with open(mask_file, 'wb') as ff:
            cPickle.dump(all_masks, ff, cPickle.HIGHEST_PROTOCOL)   

    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, output_dir)

    if cfg.TEST.MASK_ON:
        imdb.evaluate_masks(all_masks, output_dir)
