from keras.models import Sequential
from keras.layers import Convolution2D, Cropping2D, Dropout, Flatten,Dense, Lambda, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('training_data_path', './training-data',
                    "Training data directory.")
flags.DEFINE_integer('epochs', 5, "Number of epochs.")
flags.DEFINE_integer('train_batch_size', 32, "Batch size.")
flags.DEFINE_integer('test_batch_size', 1, "Batch size.")
flags.DEFINE_string('model','nvidia',
                    'Name of the model to use either lenet5 or nvidia (case-insensitive).')
flags.DEFINE_string('output_suffix','','Suffix added to saved model filename.')



__DEBUG__ = False


def load_logs(base_dir,all_camera=False, offset_correction=0.2):
    """ 
    Loads driving logs for .csv file. base_dir must be a parent directory
    containing driving_log.csv and a child directory named IMG that has all
    training images.
    """
    logfile_path = os.path.join(base_dir,'driving_log.csv')

    if all_camera:
        names = ['center', 'left', 'right', 'steering']
        load_cols = [0,1,2,3]
        logs = pd.read_csv(logfile_path, sep=',', header=None, names=names, usecols=load_cols)
        logs = pd.melt(logs, id_vars=['steering'], var_name='camera', value_name='img_path')

        # Steering correction for side cameras.
        logs['steering'].loc[logs['camera'] == 'left'] += offset_correction
        logs['steering'].loc[logs['camera'] == 'right'] -= offset_correction

    else:
        names = ['center','steering']
        load_cols = [0,3]
        logs = pd.read_csv(logfile_path, sep=',', header=None, names=names, usecols=load_cols)
        logs = pd.melt(logs, id_vars=['steering'], var_name='camera', value_name='img_path')

    print(logs.head(3))
    print("{} samples found in {}".format(len(logs),logfile_path))

    return logs


def drop_zero_steering(dataframe, drop_zero_prob=0.5, drop_range=0.1):

    orig_count = len(dataframe)

    near_zero_idx = np.where(abs(pd.to_numeric(dataframe['steering'])) < drop_range)[0]

    delete_count = int(drop_zero_prob * len(near_zero_idx))

    delete_indices = np.random.choice(near_zero_idx, delete_count, replace=False)

    # Update steering angles array.
    dataframe.drop(dataframe.index[delete_indices],inplace=True)

    new_count = len(dataframe)

    print("{} rows dropped.".format(orig_count-new_count))

    return dataframe


def process_logs(data_dir, dict_options=None):

    if dict_options is None:
        dict_options = {}
        print("Warning: No options passed to {}() defaults will be used"
              .format(data_generator.__name__))

        logs = load_logs(data_dir)
        logs = drop_zero_steering(logs)

    else:

        if dict_options.get('all_camera'):

            if dict_options.get('steering_correction') is None:

                logs = load_logs(data_dir,
                                 dict_options.get('all_camera'))

            else:
                logs = load_logs(data_dir,
                                 dict_options.get('all_camera'),
                                 dict_options.get('steering_correction'))

        else:
            if dict_options.get('steering_correction') is not None:
                print("Warning option 'steering_correction' is unused.")


        if dict_options.get('drop_zero_prob'):

            if dict_options.get('drop_zero_range'):

                logs = drop_zero_steering(logs,
                                          dict_options['drop_zero_prob'],
                                          dict_options['drop_zero_range'])
            else:

                logs = drop_zero_steering(logs,
                                          dict_options['drop_zero_prob'])
        else:
            logs = drop_zero_steering(logs)

    # Shuffle and Split into train, validation and test sets.
    train_test_ratio = \
        dict_options['train_test_ratio'] if dict_options.get('train_test_ratio') else 0.8

    logs = shuffle(logs)

    train_logs, test_logs = train_test_split(logs,train_size=train_test_ratio)
    train_logs, valid_logs = train_test_split(train_logs,train_size=0.8)

    print("Split data into {} training and {} test samples".
          format(len(train_logs),len(test_logs)))

    return (train_logs, valid_logs, test_logs)


def data_generator(data_dir,logs,dict_options=None):

    def _g():

        n_examples = len(logs)
        batch_sz = dict_options['batch_sz'] if dict_options.get('batch_sz') else 32
        augment_flipped = dict_options.get('augment_flipped')

        while True:
            n_samples = (batch_sz // 2) if augment_flipped else batch_sz

            idx = np.random.randint(0,n_examples,n_samples)
            batch = logs[['steering','img_path']].iloc[idx]

            img_dir = os.path.join(data_dir,'IMG/')
            image_data = []

            for path in batch['img_path']:
                file_name = path.split('/')[-1]
                img = cv2.cvtColor(cv2.imread(img_dir + file_name), cv2.COLOR_BGR2RGB)
                image_data.append(img)

            angles = batch['steering']

            image_data = np.array(image_data)
            angles = np.array(angles)

            if augment_flipped:
                image_data = np.vstack((image_data,np.fliplr(image_data)))
                angles = np.hstack((angles,-angles))

            yield (image_data,angles)

    return _g;



def Nvidia(input_shape):
    model = Sequential()

    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=input_shape))

    model.add(Cropping2D(((70, 25), (0, 0))))

    model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    model.add(Flatten())

    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    return model


def Lenet5(input_shape):
    model = Sequential()

    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=input_shape))

    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Convolution2D(16, 5, 5, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    return model


def main(_):
    train_data_path = FLAGS.training_data_path

    data_options = {'all_camera':True,
                    'steering_correction':0.1,
                    'drop_zero_prob':0.85,
                    'drop_zero_range':0.3,
                    'train_test_ratio':0.7
                   }
    train_logs, validation_logs, test_logs = \
        process_logs(train_data_path,dict_options=data_options)

    if __DEBUG__:
        print("Using data options:\n{}".format(data_options))
        import matplotlib.pyplot as plt
        plt.hist(train_logs['steering'], bins=np.linspace(-1, 1, 20))
        plt.title('Steering Angle Distribution')
        plt.show()

    # Create model.
    if FLAGS.model.lower() == 'nvidia':
        model = Nvidia([160,320,3])
    elif FLAGS.model.lower() == 'lenet5':
        model = Lenet5([160,320,3])
    else:
        raise Exception('Not a valid model {}.'.format(FLAGS.model))

    # Train the model using a generator.
    train_options = { 'batch_sz':FLAGS.train_batch_size,
                      'augment_flipped':True }
    validation_options =  { 'batch_sz':FLAGS.train_batch_size,
                            'augment_flipped':False }

    train_generator = data_generator(train_data_path,train_logs, train_options)
    validation_generator = data_generator(train_data_path, validation_logs,
                                          validation_options)

    nb_epochs = FLAGS.epochs
    model.fit_generator(train_generator(),
                        len(train_logs),
                        nb_epochs,
                        validation_data=validation_generator(),
                        nb_val_samples=len(validation_logs),
                        verbose=2)

    # Test the model.
    test_options = { 'batch_sz':FLAGS.test_batch_size,
                     'augment_flipped':False }
    test_generator = data_generator(train_data_path,test_logs,test_options)
    test_loss = model.evaluate_generator(test_generator(),len(test_logs),
                                         verbose=2)
    print("Test loss: {:.3f}".format(test_loss))

    # Save model to file.
    suffix = '-'+FLAGS.output_suffix if FLAGS.output_suffix != '' else ''
    model.save(FLAGS.model+suffix+'.h5')

    if __DEBUG__:
        for i in range(10):
            imgdata,label = test_generator.next()
            plt.imshow(imgdata)
            plt.title("Predicted angle {:.3f} actual angle {:.3f}".
                      format(model.predict(imgdata, batch_size=1,verbose=2)))
            plt.show()

    if __DEBUG__:

        if os.path.isdir('./snapshots'):

            filenames = os.listdir('./snapshots')

            if not os.path.isdir('./snapshots-predictions'):
                os.mkdir('./snapshots-predictions')

            for filename in filenames:
                imgdata = cv2.cvtColor(cv2.imread('./snapshots/' + filename),
                                       cv2.COLOR_BGR2RGB)
                prediction = model.predict(imgdata[None, :, :, :], batch_size=1,verbose=2)
                cv2.putText(imgdata, str(prediction), (0, 100), cv2.FONT_HERSHEY_PLAIN, 1.0, 0)
                cv2.imwrite('./snapshots-predictions/' + filename, imgdata)


if __name__ == '__main__':
    tf.app.run()
