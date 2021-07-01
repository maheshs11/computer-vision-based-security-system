
import detectron2
import cv2
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from d2go.runner import Detectron2GoRunner
from matplotlib import pyplot as plt
#from google.colab.patches import cv2_imshow
from d2go.utils.demo_predictor import DemoPredictor


runner = Detectron2GoRunner()
cfg = runner.get_default_cfg()

#cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")

#cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"))
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml")

cfg.merge_from_file(r"/content/drive/MyDrive/d2go/config.yml")
cfg.MODEL.WEIGHTS =  os.path.join("/content/drive/MyDrive/d2go/model_final.pth")
cfg.MODEL.DEVICE = "cpu"

#cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml"))
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
predictor = DefaultPredictor(cfg)
#model = runner.build_model(cfg)
#predictor = DemoPredictor(model)

#cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture("/content/drive/MyDrive/d2go/VID_20210701_130909.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

#out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))
#out = cv2.VideoWriter('output3.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
#out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (frame_width,frame_height))
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame_width,frame_height))

"""out = cv2.VideoWriter(
                'newvideo1.mkv',
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                #fourcc=cv2.VideoWriter_fourcc('M','J','P','G'), 
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )"""

while (cap.isOpened()):
	
    ret,frame=cap.read(0)
    print(fps)
    print(num_frames)
    
    try:
      outputs = predictor(frame)
      #v = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
      v = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.IMAGE_BW)
      v = v.draw_instance_predictions(frame, outputs["instances"].to('cpu'))
      #cv2_imshow("window", v.get_image())
      
      out.write(v.get_image())
      
      print(outputs["instances"].pred_boxes)
      print(outputs["instances"].pred_classes)
      omt = str(outputs["instances"].pred_classes)
      outpred = omt[8:9]
      print(outpred)
     
    
    except:
      break
    
cap.release()
out.release()
#cv2.destroyAllWindows()
