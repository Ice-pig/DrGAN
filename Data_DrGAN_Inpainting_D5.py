
import numpy as np
import time


from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Input, Dense, Reshape, Flatten, concatenate
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Conv1D, MaxPooling1D,UpSampling1D
from keras.layers import LeakyReLU, Dropout


from keras.optimizers import Adam, RMSprop, SGD
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model

from keras.layers.advanced_activations import LeakyReLU,ELU

from keras import regularizers
from keras import initializers
from keras.layers import LSTM

import datetime
import scipy.io as scio
import random

from sklearn import metrics

from keras.models import load_model

from keras import backend as K
import tensorflow as tf
from icepig import test_score_program
from icepig import icepig_dataload,icepig_memery


class GAN():
    def __init__(self):
        self.img_rows = 1
        self.img_cols = 26

        self.miss_dim = 3

        self.channels = 1
        self.latent_dim = (self.img_cols - self.miss_dim) # generator中的输入维度
        self.aim_output_dim = self.miss_dim
        self.fc_units = 500


        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.data_shape = self.img_cols
        self.fakedata_shape = self.latent_dim

        self.D_model_output_dim = 1

        #-------CNN全局变量----------------

        self.filters_1, self.kernel_size_1, self.pool_size_1 = 20, 3, 2
        self.filters_2, self.kernel_size_2, self.pool_size_2 = 10, 2, 2
        self.filters_3, self.kernel_size_3, self.pool_size_3 = 5, 2, 2
        self.filters_4, self.kernel_size_4, self.pool_size_4 = 5, 2, 2

        #-------------------------------------

        self.MPL_trained_model = load_model('F:\\doctor_progress\\model_trained\\Model_D1_D6\\DNN_D5.h5')
        self.time_lable = datetime.datetime.now().strftime('%Y-%m-%d')
        self.min_rmse = 0.3
        self.timestep = 1
        momentum = 0.9

        D_learning_rate = 5e-4
        G_learning_rate = 5e-4

        optimizer_D = RMSprop(lr=D_learning_rate)
        optimizer_G = RMSprop(lr=G_learning_rate)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer_D,
                                   metrics=['accuracy'])

        # =========Build the generator===============================
        self.generator = self.build_generator()

        input1_A = Input(shape=(self.latent_dim,), name='input1')
        input2_B = Input(shape=(self.latent_dim,), name='input2')

        fake_data = self.generator(inputs=[input1_A, input2_B])

        self.discriminator.trainable = False
        validity = self.discriminator(fake_data)

        self.GAN_model = Model([input1_A, input2_B], validity, name = 'GAN_model')
        self.GAN_model.compile(loss='binary_crossentropy', optimizer=optimizer_G )

    def test_program(self, y_true, y_prediction):  # 测试程序
        y_true = y_true
        y_predict = y_prediction
        MSE = metrics.mean_squared_error(y_true, y_predict)
        RMSE = np.sqrt(metrics.mean_squared_error(y_true, y_predict))
        R2 = metrics.r2_score(y_true, y_predict)
        MEA = metrics.mean_absolute_error(y_true, y_predict)

        return (MSE, RMSE, R2, MEA)

    def build_generator(self):

        input1_ = Input(shape=(self.fakedata_shape,), name='input1')
        input2_ = Input(shape=(self.fakedata_shape,), name='input2')

        x1 = input1_

        cov_shape = (self.fakedata_shape, self.channels)
        x1 = Reshape(target_shape=cov_shape)(x1)

        # 卷积 CNN_layer_1
        x1 = UpSampling1D(size=self.pool_size_3)(x1)
        x1 = Conv1D(filters=self.filters_3, kernel_size=self.kernel_size_3, strides=1, padding="same")(x1)
        x1 = BatchNormalization(momentum=0.8)(x1)
        x1 = Activation("relu")(x1)
        # 卷积 CNN_layer_2
        x1 = UpSampling1D(size=self.pool_size_2)(x1)
        x1 = Conv1D(filters=self.filters_2, kernel_size=self.kernel_size_2, strides=1, padding="same")(x1)
        x1 = BatchNormalization(momentum=0.8)(x1)
        x1 = Activation("relu")(x1)
        # 卷积 CNN_layer_3
        x1 = UpSampling1D(size=self.pool_size_1)(x1)
        x1 = Conv1D(filters=self.filters_1, kernel_size=self.kernel_size_1, strides=1, padding="same")(x1)
        x1 = Activation("tanh")(x1)

        x1 = Flatten()(x1)

        x1 = Dense(units=self.fc_units, activation=None, use_bias=True,
                   kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                   bias_initializer='zeros',
                   kernel_regularizer=regularizers.l1_l2(0.00), bias_regularizer=None,
                   activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)

        x1 = Dense(units=self.fc_units, activation=None, use_bias=True,
                   kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                   bias_initializer='zeros',
                   kernel_regularizer=regularizers.l1_l2(0.00), bias_regularizer=None,
                   activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)

        x1 = Dense(units=self.aim_output_dim, activation=None, use_bias=None,
                   kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None),
                   bias_initializer='zeros',
                   kernel_regularizer=regularizers.l1_l2(0.00), bias_regularizer=None,
                   activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x1)
        x1 = Activation('tanh')(x1)

        x2 = input2_
        x = concatenate([x2, x1])
        # x = Reshape(self.data_shape)(x)
        output_ = x

        model = Model(inputs=[input1_, input2_], outputs=[output_], name='Generator_model')
        model.summary()

        return model

    def build_discriminator(self):

        input = Input(shape=(self.data_shape,))  # 这里必须有个逗号，变成数组结构

        cov_shape = (self.data_shape, self.channels)
        x1 = Reshape(target_shape=cov_shape)(input)

        x1 = Conv1D(filters=self.filters_1, kernel_size=self.kernel_size_1, strides=1, )(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = Dropout(rate=0.25)(x1)

        x1 = Conv1D(filters=self.filters_2, kernel_size=self.kernel_size_2, strides=1, )(x1)
        x1 = BatchNormalization(momentum=0.8)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = Dropout(rate=0.25)(x1)

        x1 = Conv1D(filters=self.filters_3, kernel_size=self.kernel_size_3, strides=1, )(x1)
        x1 = BatchNormalization(momentum=0.8)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = Dropout(rate=0.25)(x1)

        x1 = Conv1D(filters=self.filters_4, kernel_size=self.kernel_size_4, strides=1, )(x1)
        x1 = BatchNormalization(momentum=0.8)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = Dropout(rate=0.25)(x1)

        x1 = Flatten()(x1)

        x1 = Dense(units=self.fc_units, activation=None, use_bias=True,
                   kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                   bias_initializer='zeros',
                   kernel_regularizer=regularizers.l1_l2(0.00), bias_regularizer=None,
                   activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)

        x1 = Dense(units=self.fc_units, activation=None, use_bias=True,
                   kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                   bias_initializer='zeros',
                   kernel_regularizer=regularizers.l1_l2(0.00), bias_regularizer=None,
                   activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)

        x1 = Dense(units=self.D_model_output_dim, activation=None, use_bias=None,
                   kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None),
                   bias_initializer='zeros',
                   kernel_regularizer=regularizers.l1_l2(0.00), bias_regularizer=None,
                   activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x1)
        x1 = Activation("sigmoid")(x1)

        output = x1

        model = Model(inputs=input, outputs=output, name='Discriminator_model')
        model.summary()

        return model





    def train(self, epochs, batch_size):

        # Load the dataset
        min_rmse =  self.min_rmse

        datasets_filemath = 'F:\\doctor_progress\\Datasets_icepig\\paper_data\\D5.mat'

        dataunit = icepig_dataload.Dataload_v3_MinMax(
            datasets_filemath, train_size=0.2, test_size=0.02, random_state=9527)
        x_train, y_train, x_test, y_test = dataunit[0], dataunit[1], dataunit[2], dataunit[3],

        self.new_train = x_train
        self.new_y_train = y_train

        # Adversarial ground truths
        valid = np.ones(    (batch_size, 1)    )    #告诉判别器都是 1
        fake = np.zeros(    (batch_size, 1)    )    ##告诉判别器都是 0


        train_test_data = []
        best_RMSE = []
        y_pre_unit = []
        memery_unit = []
        time_start = time.clock()

        plot_model(self.generator, to_file='.\\SAVE_D5\\GAN_model\\GAN-generator-' + '.png',
                   show_shapes=True, show_layer_names=True, rankdir='LR')

        plot_model(self.discriminator, to_file='.\\SAVE_D5\\GAN_model\\GAN-Discriminator-' + str(self.time_lable) + '.png',
                   show_shapes=True, show_layer_names=True, rankdir='LR')

        plot_model(self.GAN_model, to_file='.\\SAVE_D5\\GAN_model\\GAN-Structure-' + str(self.time_lable) + '.png',
                   show_shapes=True, show_layer_names=False, rankdir='LR')

        for i in range(epochs):

            idx_01 = np.random.randint(0, self.new_train.shape[0], size=batch_size)
            imgs_train_01 = self.new_train[idx_01]
            noise_01 = self.new_train[idx_01, 0:self.latent_dim]

            # Generate a batch of new images
            fake_imgs = self.generator.predict([noise_01, noise_01])

            x_true_fake = np.concatenate([imgs_train_01, fake_imgs])
            y_true_fake_label = np.concatenate([valid, fake])

            # Train the discriminator
            d_loss = self.discriminator.train_on_batch(x_true_fake, y_true_fake_label)

            # ------------------------------------------------------
            #                        Train Generator
            # -----------------------------------------------------
            idx_02 = np.random.randint(0, self.new_train.shape[0], size=batch_size)
            noise_02 = self.new_train[idx_02, 0:self.latent_dim]
            g_loss = self.GAN_model.train_on_batch(  [noise_02, noise_02], valid  )

            #=====================测试test========================================

            y_train_son = self.new_y_train[idx_02]      #抽样train_lable
            x_test_son = x_test[:, 0:self.latent_dim]

            time_now = time.clock()
            time_consume = time_now - time_start
            lasttime =  (time_consume * (epochs - i) ) / (3600 * (i+0.1) )


            if i % 100 == 0:
                #训练伪数据生成
                x_pre_train = self.generator.predict([noise_02, noise_02])
                #x_pre_train= x_pre_train.reshape( (-1, self.timestep, self.img_cols) )
                y_pre_train = self.MPL_trained_model.predict(x_pre_train)

                #测试伪数据生成
                x_pre_test = self.generator.predict([x_test_son, x_test_son])
                #x_pre_test = x_pre_test.reshape((-1, self.timestep, self.img_cols))
                y_pre_test = self.MPL_trained_model.predict(x_pre_test)

                MSE_train, RMSE_train, MEA_train, R2_train, EVS_train = test_score_program.score_calculation(y_train_son, y_pre_train)
                MSE_test, RMSE_test, MEA_test, R2_test, EVS_test = test_score_program.score_calculation(y_test, y_pre_test)

                unit_test = [i, MSE_train, RMSE_train, MEA_train, R2_train, EVS_train, MSE_test, RMSE_test, MEA_test, R2_test, EVS_test, d_loss[0], g_loss, time_consume]
                train_test_data.append(unit_test)
                scio.savemat('.\\SAVE_D5\\savemat\\train-test-data-' + str(self.time_lable) + '.mat',
                             {'train_test_data': train_test_data })

                #print(self.mul_model.metrics_names)

                print("%d [D loss: %f, acc.: %.3f%%] [G loss: %.3f]" % (i, d_loss[0], 100 * d_loss[1], g_loss))
                print("\r train RMSE  %.5f, test RMSE %.5f, Time_remaining = %.2f h \n" % (RMSE_train, RMSE_test, lasttime))

                # ---------------------      memery ---------- -------------------------
                memery_data = icepig_memery.memery_info()
                memery_unit.append(memery_data)
                scio.savemat('.\\SAVE_D5\\savemat\\memery-data-' + str(self.time_lable) + '.mat',
                             {'memery_unit': memery_unit})



                if RMSE_test < min_rmse:
                    best_unit = [i, MSE_train, RMSE_train, MEA_train, R2_train, EVS_train, MSE_test, RMSE_test, MEA_test, R2_test,EVS_test, d_loss[0], g_loss, time_consume]
                    best_RMSE.append(best_unit)

                    scio.savemat('.\\SAVE_D5\\savemat\\best-data-' + str(self.time_lable) + '.mat',   {'best_RMSE': best_RMSE} )

                    self.generator.save('.\\SAVE_D5\\GAN_model\\FC-model-generator-' + str(RMSE_test) + '-' + str(i) + '-' + str(
                        self.time_lable) + '.h5')
                    y_pre_unit.append(y_pre_test)
                    min_rmse = RMSE_test-0.001

                scio.savemat('.\\SAVE_D5\\savemat\\y_pre_data-' + str(self.time_lable) + '.mat',
                             {'y_pre_unit': y_pre_unit})

        #=======               =SAVE=                  ===========
        self.discriminator.save('.\\SAVE_D5\\GAN_model\\FC-model-discriminator-' + str(self.time_lable) + '.h5')
        self.generator.save('.\\SAVE_D5\\GAN_model\\FC-model-generator-' + str(self.time_lable) + '.h5')
        self.GAN_model.save('.\\SAVE_D5\\GAN_model\\FC-model-GAN_model-' + str(self.time_lable) + '.h5')



def run_train(Epoches=20*1000, Size=500 ):
    gan = GAN()
    gan.train(epochs=Epoches, batch_size=Size)
    # 使用完模型之后，清空之前model占用的内存
    K.clear_session()
    tf.reset_default_graph()
    return()



if __name__ == '__main__':
    run_train()

