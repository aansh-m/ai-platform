from imports import *
from data_processing import *
from model import *

keys = 'train_path','batch_size','epochs'
args = {k:v for k,v in zip(keys,sys.argv[1:])}
print(args)

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = args['train_path']
epochs = int(args['epochs'])
batch_size = int(args['batch_size'])


X_train = image_processing(IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS,TRAIN_PATH)
Y_train = mask_processing(IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS,TRAIN_PATH)


with mlflow.start_run():
    #Paste all code here
    model = build_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
    model.summary()

    model_path = "./TrainedModelsUNET/nuclei_finder_unet_1.h5"
    checkpoint = ModelCheckpoint(model_path,
                                 monitor="val_loss",
                                 mode="min",
                                 save_best_only=True,
                                 verbose=1)

    earlystop = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=1,
                              restore_best_weights=True)

    results = model.fit(X_train, Y_train, validation_split=0.1,
                        batch_size=batch_size, epochs=epochs,
                        callbacks=[earlystop, checkpoint])
    mlflow.log_param("mean_iou", mean_iou)
    mlflow.keras.log_model(model, "models")
    #mlflow.keras.save_model(model, model_path)




