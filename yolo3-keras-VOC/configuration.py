
TRAINDATA_trainsfer = True  # coco data 학습된 yolov3 darknet 가중치 로딩

YOLO_V3_WEIGHTS = "./checkpoints/yolov3.weights"

YOLO_COCO_CLASSES           = "./dataset/coco.names"

YOLO_STRIDES                = [8, 16, 32]
YOLO_IOU_LOSS_THRESH        = 0.5
YOLO_ANCHOR_PER_SCALE       = 3
YOLO_MAX_BBOX_PER_SCALE     = 100
YOLO_INPUT_SIZE             = 416

YOLO_ANCHORS = [[[10, 13], [16, 30], [33, 23]],        # 8 stride 시 앵커 (small obj dectect)
                [[30, 61], [62, 45], [59, 119]],       # 16 stride 시 앵커(midium obj dectect)
                [[116, 90], [156, 198], [373, 326]]]   # 32 stride 시 앵커(large obj dectect)

# Trainset Options
# 원본 voc2012 dataset 은 D:\딥러닝_객체인식_2022\yolov3_학습자료\VOC2012_원본 참조
# 본 예제는 train 과정을 단축해서 보여주기 위해 train dataset 줄임
TRAIN_CLASSES    =  "./dataset/voc/voc2012.names"
TRAIN_ANNOT_PATH =  "./dataset/voc/voc2012_train.txt"
TRAIN_CHECKPOINTS_FOLDER    = "./checkpoints"
DATA_TYPE = "yolo_v3_voc"
TRAIN_BATCH_SIZE            = 1 # 원래는 16 --> 1로 수정해 테스트중
TRAIN_INPUT_SIZE            = 416
TRAIN_DATA_AUG              = True
TRAIN_LOAD_IMAGES_TO_RAM    = True # With True faster training, but need more RAM

TRAIN_LR_INIT               = 1e-4
TRAIN_LR_END                = 1e-6
TRAIN_WARMUP_EPOCHS         = 1
TRAIN_EPOCHS                = 4

YOLO_STRIDES                = [8, 16, 32]
# YOLO_IOU_LOSS_THRESH        = 0.5
# YOLO_ANCHOR_PER_SCALE       = 3
# YOLO_MAX_BBOX_PER_SCALE     = 100

# TESTset options
TEST_ANNOT_PATH  =  "./dataset/voc/voc2012_val.txt"
TEST_BATCH_SIZE             = 16
TEST_INPUT_SIZE             = 416
TEST_DATA_AUG               = False


SIZE_TRAIN = 512*TRAIN_BATCH_SIZE
SIZE_TEST  = 128*TEST_BATCH_SIZE
