# -*- coding: utf-8 -*-

class Config(object):

    # Folders

    TRAIN_DIR = '../ttsr/ttsr_packed_png_128px_train'
    #'/rds/user/mb2139/hpc-work/ttsr_packed_png_128px_train/'

    VAL_DIR = '../ttsr/ttsr_packed_png_128px_val'
    #'/rds/user/mb2139/hpc-work/ttsr_packed_png_128px_val/'

    LOG_DIR = 'logs/'
    #'/rds/user/mb2139/hpc-work/DRA/logs/'

    CHK_DIR = 'saved_weights/'
    #'/rds/user/mb2139/hpc-work/DRA/checkpoints/'

    ROOT_DIR = './'
    #'/home/mb2139/DRA/'

    CONFIG_DIR = 'configs/'
    #'/rds/user/mb2139/hpc-work/DRA/configs/'

    # Augmentation

    MAX_ANGLE = 180

    TRANSLATE = (0.,0.)

    SCALE = (.9,1.1)

    SHEAR = (-.2,.2,-.2,.2)

    BRIGHTNESS = .3

    CONTRAST = [100,500]

    POINT_PROB = .02

    POINT_SIGMA = [.3,.8]

    EXPOSURE_RANGE = [1,30]

    GAUSS_MAX_MEAN = 10

    GAUSS_MAX_STD = 5

    # Training

    RESUME_FROM = 'run20221024T1438'

    EPOCHS = 2 # 100

    BACKUP_RATE = 1 # 10

    BATCH_SIZE = 4 # 8

    CHARBONNIER_EPSILON = 1e-3 # 1e-3

    LEARNING_RATE = 2e-4 # 2e-4

    WEIGHT_DECAY = 2e-2 # 2e-2

    SCHEDULING_STEP_SIZE = 20

    SCHEDULING_GAMMA = .75

    # Model

    IMG_DIM = 128

    IMG_CH = 1

    PROJ_DIM = 16 # 32

    WINDOW_SIZE = 8 # 8

    ATTN_HEADS = [1,2,4,8]

    ATTN_DIM = 32

    DROPOUT = .1

    LEFF_FILTERS = 32 # 32

    STAGES = [2,2,2,2]  #[2,2,2,2]
