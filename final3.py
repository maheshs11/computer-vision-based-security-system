
import detectron2
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from lockit import lock , unlock


runner = Detectron2GoRunner()
cfg = runner.get_default_cfg()
cfg.MODEL.DEVICE = "cuda" #"cpu"#

#cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")

#cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"))
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml")

cfg.merge_from_file(r"/content/drive/MyDrive/d2go/config.yml")
cfg.MODEL.WEIGHTS =  os.path.join("/content/drive/MyDrive/d2go/model_final.pth")

#cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml"))
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model
predictor = DefaultPredictor(cfg)


#cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture("video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

#out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))
#out = cv2.VideoWriter('output1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))


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
      print(outputs["instances"].pred_classes)
      omt = str(outputs["instances"].pred_classes)
      outpclass = omt[8:9]
      print(outpclass)
      if outpclass == '0': # class 0
                        
                        '''
                        Here we using solenoid lock to for lock-unlock the door,
                        Lock connected with relay module and relay module is set on 26 pin(BCM mode) of Pi.
                        '''
                        unlock(26)
                        time.sleep(10) #Lock will remains open for 10 seconds.
                        lock(26)
                        GPIO.cleanup(26)
      
      #out.write(v.get_image())
    #cv2_imshow("Moda", v.get_image())
    except:
      break
    
cap.release()
out.release()
#cv2.destroyAllWindows()
