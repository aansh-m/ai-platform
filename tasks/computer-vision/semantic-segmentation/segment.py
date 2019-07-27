from imports import *
from data_processing import *
from model import mean_iou

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = './data/sample/'

keys = 'input_path','model_path'
args = {k:v for k,v in zip(keys,sys.argv[1:])}


img = imread(args['input_path'])[:, :, :IMG_CHANNELS]
img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

model = load_model(args['model_path'], custom_objects={'mean_iou': mean_iou})

preds_train = model.predict(np.array([img],), verbose=1)
preds_train_t = (preds_train > 0.5).astype(np.uint8)

ix = random.randint(0, 5)
plt.figure(figsize=(15,15))

plt.subplot(121)
imshow(args['input_path'])
plt.title("Input Image")

plt.subplot(122)
imshow(np.squeeze(preds_train_t > 0.5))
plt.title("Segmented")
plt.show()
