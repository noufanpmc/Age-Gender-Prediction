import os

# APP_PATH = "C:\\flask\\age-gender"
APP_PATH = "C:\\Users\\t_shivamth\\Desktop\\age_gender\\190221\\code"
IMAGE_PATH = os.path.join(APP_PATH, "img.jpg")
AGE_WEIGHTS = os.path.join(APP_PATH, "age_weights\\ssrnet_pretrained.h5")
GENDER_CHECKPOINT = os.path.join(APP_PATH, "gender_checkpoints")