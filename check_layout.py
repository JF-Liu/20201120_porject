from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.layers import Dense,GlobalAveragePooling2D,BatchNormalization
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import backend as K
from keras.models import Sequential, Model
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def feature_extraction_InV3(img_width, img_height,
                        train_data_dir,
                        num_image,
                        epochs):
    base_model = InceptionV3(input_shape=(299, 299, 3),
                              weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    model = Model(inputs=base_model.input, outputs=x)


    train_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(train_data_dir,
    target_size = (299, 299),
    batch_size = 15,
    class_mode = "categorical",
    shuffle=False)

    y_train=train_generator.classes
    y_train1 = np.zeros((num_image, 2))
    y_train1[np.arange(num_image), y_train] = 1

    train_generator.reset
    X_train=model.predict_generator(train_generator,verbose=1)
    print (X_train.shape,y_train1.shape)
    return X_train,y_train1,model

def train_last_layer(img_width, img_height,
                        train_data_dir,
                        num_image,
                        epochs = 50):


    X_train,y_train,model=feature_extraction_InV3(img_width, img_height,
                            train_data_dir,
                            num_image,
                            epochs)

    X_test,y_test,model=feature_extraction_InV3(img_width,img_height,
                            test_data_dir,
                            num_test_image,
                            epochs)

    my_model = Sequential()
    my_model.add(BatchNormalization(input_shape=X_train.shape[1:]))
    my_model.add(Dense(1024, activation = "relu"))
    my_model.add(Dense(2, activation='softmax'))
    my_model.compile(optimizer="SGD", loss='categorical_crossentropy',metrics=['accuracy'])
    #early = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    # my_model.fit(X_train, y_train,epochs=18,batch_size=30,verbose=1)
    history = my_model.fit(X_train, y_train,epochs=18,
                 validation_data=(X_test,y_test),
                 # validation_split=0.2, #切割20%訓練集做validataion
                 # shuffle=1,
                 batch_size=30,verbose=1)
    # plot_training(history,'./')
    my_model.save('./binary_for_style_ver2.h5')
    return history

def plot_training(history_ft):
    print(history_ft.history.keys())
    acc = history_ft.history['accuracy']
    val_acc = history_ft.history['val_accuracy']
    loss  = history_ft.history['loss']
    val_loss = history_ft.history['val_loss']
    epoches = range(len(acc))
    plt.plot(epoches,acc,'r',color ='green',label = 'acc')
    plt.plot(epoches,val_acc,'--r',label = 'val_acc') #default color
    plt.title('Training and Validataion accuracy')
    plt.figure()
    plt.plot(epoches,loss,'r',color = 'green',label = 'loss')
    plt.plot(epoches,val_loss,'--r',label = 'val_loss')
    plt.title('Training and Validataion loss')
    plt.savefig('path')  # Save the current figure.plt.savefig('path') #Save the current figure.
    plt.show()
    print('acc:',acc)
    print('val_acc:',val_acc)
    print('epoches:',range(len(acc)))
    print('loss:',loss)
    print('val_loss',val_loss)

if __name__=="__main__":
    img_width=299
    img_height = 299
    train_data_dir = "./train"
    test_data_dir = "./test"
    # num_image=4309
    num_image=4309 #Train的圖片總數
    num_test_image = 600 #Test的圖片總數
    epochs = 10
    model=train_last_layer(img_width, img_height,
                            train_data_dir,
                            num_image,epochs)

    plot_training(model)