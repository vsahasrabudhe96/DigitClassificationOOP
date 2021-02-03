
'''
The preprocess package has all the preprocessing steps
'''
def main():
    from preprocess import MNIST
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    import tensorflow as tf
    from keras.models import load_model
    from model import Model
    import pickle
    import joblib
    import os
    import warnings
    import matplotlib.pyplot as plt
    import numpy as np
    warnings.filterwarnings("ignore", category=FutureWarning)
    dataset = MNIST()
    model = Model()
    dataset.data_load()
    print('dataset loaded successfully')
    dataset.split_X_y()
    dataset.prep_test()
    print(dataset.X_train.shape)
    print(dataset.y_train.shape)
    dataset.reshape(train=True)
    print(dataset.X_train.shape)
    dataset.reshape(test=True)
    print(dataset.X_test.shape)
    # print('plotting test data')
    # dataset.plot(2,test=True)

    INPUT_SHAPE= dataset.X_train.shape[1:]
    OUTDIM = dataset.train_df['label'].nunique()
    EPOCHS = 50
    BATCH_SIZE=64
    VAL_SPLIT=0.1
    if os.path.exists('my_model.h5'):
        print('MODEL EXISTS')
        print('-------LOADING MODEL---------')

        load_model = load_model('my_model.h5')
        y_predicted = model.predict(load_model, dataset.X_test)
    else:
        print('MODEL DOES NOT EXISTS')
        print('-------TRAINING THE MODEL--------')
        model.build_model(INPUT_SHAPE,OUTDIM)
        history, trained_model = model.train_model(EPOCHS,BATCH_SIZE,dataset.X_train ,dataset.y_train ,VAL_SPLIT)
        
        trained_model.save("my_model.h5")
        y_predicted = model.predict(trained_model, dataset.X_test)
        
    plt.figure(figsize=(15,7))
    for i in range(8):
        
        plt.subplot(3,3,i+1)
        plt.imshow(dataset.X_test[i][:,:,-1])
        plt.title('Predicted Label: {}'.format(np.argmax(y_predicted[i])))
    plt.show()

if __name__ == "__main__":
   main()