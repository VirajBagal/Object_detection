# Object_detection
In this repo, I compare the performance of FasterRCNN with ResNet50 backbone and EfficientDet-B5 on wheat head multi-object detection.
The wheat head dataset can be obtained from here: https://www.kaggle.com/c/global-wheat-detection

# Train

```python
python train.py --run_name give_run_name --dir_input path_to_csv --dir_train path_to_images --num_classes num_of_classes_in_dataset
```

# Qualitative comparison

Blue boxes are the predictions from Faster-RCNN, while red are ones from EfficientDet-B5. We see that at some places, Faster-RCNN predicts larger boxes than EfficientDet-B5. At left mid spot, Faster-RCNN predicts box but EfficientDet-B5 doesn't and we see that indeed there is no wheat head over there. 


![alt text](https://github.com/VirajBagal/Object_detection/blob/main/eval.png?raw=true)

# Quantitative comparison

We see that EfficientDet-B5 has lesser number of trainable parameters than Faster-RCNN. EfficientDet-B5 also trains faster than Faster-RCNN. Moreover, it also has better public and private test score than Faster-RCNN. 

![alt text](https://github.com/VirajBagal/Object_detection/blob/main/obj_detection.png?raw=true)
