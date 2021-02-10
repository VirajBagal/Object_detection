from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.loss import DetectionLoss
from effdet.anchors import Anchors, AnchorLabeler
from effdet.efficientdet import HeadNet
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from utils import calculate_image_precision




class EfficientDetModel(pl.LightningModule):
    def __init__(self, config):
        super(Model, self).__init__()
        
        self.config = config
        
        self.model, new_config = self.get_net()
        self.detbench = DetBenchTrain(new_config)


    def forward(self, batch):

        images, targets, image_ids = batch
        
        images = torch.stack(images)
        boxes = [t['boxes'].float() for t in targets]
        labels = [t['labels'].float() for t in targets]
        class_score, box_score = self.model(images)
        
        loss, _, _ = self.detbench(class_score, box_score, boxes, labels)

        return loss, class_score, box_score

    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr = self.config.lr)

        scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = self.config.patience, factor = self.config.factor),
                    'monitor': 'val_loss'
                    }
            
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def training_step(self, batch, batch_idx):

        loss, _, _ = self.forward(batch)
   
        self.log_dict({'train_loss': loss}, on_step = True, on_epoch = True, logger = True)
        
        return loss
    

    def validation_step(self, batch, batch_idx): 

        loss, scores, preds = self.forward(batch)
        
        self.log_dict({'val_loss': loss}, on_epoch = True, logger = True)
        
        
    def get_net(self):
        config = get_efficientdet_config('tf_efficientdet_d5')
        net = EfficientDet(config, pretrained_backbone=False)
        checkpoint = torch.load('../input/efficientdet/efficientdet_d5-ef44aea8.pth')
        net.load_state_dict(checkpoint)
        config.num_classes = 1
        config.image_size = 256
        net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
        return net, config


class DetBenchTrain(nn.Module):
    def __init__(self, config):
        super(DetBenchTrain, self).__init__()
        self.config = config
   
        anchors = Anchors(
            config.min_level, config.max_level,
            config.num_scales, config.aspect_ratios,
            config.anchor_scale, config.image_size)
        self.anchor_labeler = AnchorLabeler(anchors, config.num_classes, match_threshold=0.5)
        self.loss_fn = DetectionLoss(self.config)

    def forward(self, class_out, box_out, gt_boxes, gt_labels):

        cls_targets = []
        box_targets = []
        num_positives = []
        # FIXME this may be a bottleneck, would be faster if batched, or should be done in loader/dataset?
        for i in range(len(gt_boxes)):
            gt_class_out, gt_box_out, num_positive = self.anchor_labeler.label_anchors(gt_boxes[i], gt_labels[i])
            cls_targets.append(gt_class_out)
            box_targets.append(gt_box_out)
            num_positives.append(num_positive)

        return self.loss_fn(class_out, box_out, cls_targets, box_targets, num_positives)



class FasterRCNNModel(pl.LightningModule):
    def __init__(self, config):
        super(Model, self).__init__()
        
        self.config = config
        
        # load a model; pre-trained on COCO
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        num_classes = 2  # 1 class (wheat) + background

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.config.num_classes)

        self.iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]
        self.validation_image_precisions = []

    def forward(self, batch):

        images, targets, image_ids = batch
        
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        # separate losses
        loss_dict = self.model(images, targets)

        return loss_dict

    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr = self.config.lr)

#         scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = self.config.patience, factor = self.config.factor),
#                     'monitor': 'val_loss'
#                     }
            
        return {'optimizer': optimizer}
    
    def training_step(self, batch, batch_idx):

        loss_dict = self.forward(batch)
        loss = sum(loss for loss in loss_dict.values())/len(batch[0])
   
        self.log_dict({'train_loss': loss}, on_step = True, on_epoch = True, logger = True)
        
        return loss
    

    def validation_step(self, batch, batch_idx): 

        output = self.forward(batch)
        
        scores = torch.cat([s['scores'] for s in output]).cpu().numpy()
        preds = torch.cat([s['boxes'] for s in output]).cpu().numpy()
        gt_boxes = torch.cat([t['boxes'] for t in batch[1]]).cpu().numpy()
        
        preds_sorted_idx = np.argsort(scores)[::-1]
        preds_sorted = preds[preds_sorted_idx]

        
        image_precision = calculate_image_precision(preds_sorted,
                                                    gt_boxes,
                                                    thresholds=self.iou_thresholds,
                                                    form='pascal_voc')

        self.validation_image_precisions.append(image_precision)

        self.log_dict({'val_mAP': image_precision}, on_epoch = True, logger = True)