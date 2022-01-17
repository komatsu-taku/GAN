from http.client import METHOD_NOT_ALLOWED
from re import M
from sys import byteorder

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten, Input, LeakyReLU,
                                     Reshape, UpSampling2D)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam


class DCGAN(tf.keras.models.Model):
    def __init__(self, img_rows, img_cols, img_channels, z_dim):
        """
        コンストラクタ : 基本構造の作成
        """
        super().__init__()
        self.path = "./images"

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.img_shape = (img_rows, img_cols, img_channels)

        # 潜在変数の値
        self.z_dim = z_dim
        
        # optimizerの設定
        optimizer_gen = Adam(0.0002, 0.5)
        optimizer_dis = Adam(1e-5, beta_1=0.5)
        
        # discriminatornの設定
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss = "binary_crossentropy",
            optimizer = optimizer_dis,
            metrics = ["accuracy"]
        )

        # generatorの設定
        self.generator = self.build_generator()

        # generatorの学習のためのcombine-neroworkの設定
        self.combined = self.build_combined1()
        # self.combined2 = self.build_combined2()
        self.combined.compile(
            loss = "binary_crossentropy",
            optimizer = optimizer_gen,
            metrics = ["accuracy"]
        )

    def build_generator(self):
        """
        generatorの設定
        """
        # 潜在変数の設定
        noise_shape = (self.z_dim, )

        # モデルの構築
        model = Sequential()
        
        model.add(Dense(1024, input_shape=noise_shape))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dense(128*7*7))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Reshape((7, 7, 128), input_shape=(128*7*7, )))
        model.add(UpSampling2D((2,2)))
        model.add(Conv2D(64, (5, 5), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(UpSampling2D((2,2)))
        model.add(Conv2D(1, (5, 5), padding="same"))
        model.add(Activation("tanh"))
        
        model.summary()
        
        return model
    
    def build_discriminator(self):
        """
        discriminator の作成
        """
        img_shape = self.img_shape

        # モデルの定義
        model = Sequential()
        model.add(Conv2D(64, (5,5), (2,2), padding="same", input_shape=img_shape))
        model.add(LeakyReLU(0.2))
        model.add(Conv2D(128, (5,5), strides=(2,2)))
        model.add(LeakyReLU(0.2))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
        
        return model

    def build_combined1(self):
        """
        Generator学習用
        Generetor と Discriminator を直列に繋ぐ
        """
        self.discriminator.trainable = False
        model = Sequential([self.generator, self.discriminator])
        return model
    
    def build_combined2(self):
        z = Input(shape=(self.z_dim, ))
        img = self.generator(z)
        self.discriminator.trainable = False
        valid = self.discriminator(img)
        model = Model(z, valid)
        model.summary()

        return model
    
    def train(self, epochs, batch_size=128, save_interval=50):

        # 今回はmnistデータを用いる
        (X_train, _), (_, _) = mnist.load_data()

        # 値を-1~1に正規化
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):
            """
            Discriminatorの学習
            """

            # バッチサイズの半数をGeneratorから生成
            noise = np.random.normal(0, 1, (half_batch, self.z_dim))
            gen_imgs = self.generator.predict(noise)

            # バッチサイゼうの半数を教師データからピックアップ
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            # discriminatornの学習
            # ここでは本物のデータを偽物のデータは別々に学習させる
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            # 各損失関数の平均をとる
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            """
            Generator の学習
            """
            noise = np.random.normal(0, 1, (batch_size, self.z_dim))
            
            # 生成データのラベルは1
            valid_y = np.array([1]*batch_size)

            # 学習
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # 進捗の表示
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))

            # 指定した間隔で生成画像を保存
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
        
    # def train(self, epochs, batch_size=128, save_interval=50):
    #     """
    #     学習用の関数
    #     """
    #     # 今回はmnistのデータを用いる
    #     (X_train, _), (_, _) = mnist.load_data()

    #     # 値の正規化
    #     X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    #     X_train = np.expand_dims(X_train, axis=3)

    #     half_batch = int(batch_size / 2)

    #     for epoch in range(epochs):
    #         """
    #         Discriminatornの学習
    #         """

    #         # バッチサイズの半数をGeneratorから生成
    #         noise = np.random.normal(0, 1, (half_batch, self.z_dim))
    #         gen_imgs = self.generator.predict(noise)

    #         # 残りの半分を教師データからピックアップ
    #         idx = np.random.randint(0, X_train.shape[0], half_batch)
    #         imgs = X_train[idx]

    #         # discriminatorの学習
    #         # 別々に学習
    #         d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
    #         d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
    #         # 各損失関数の平均をとる
    #         d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    #         """
    #         Generatorの学習
    #         """
    #         noise = np.random.normal(0, 1, (batch_size, self.z_dim))

    #         # 生成ラベルは1
    #         valid_y = np.array([1] * batch_size)

    #         # 学習
    #         g_loss = self.combined.train_on_batch(noise, valid_y)
            
    #         # 進捗の表示
    #         print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))
            
    #     # 指定した間隔で生成画像を保存
    #     if epoch % save_interval == 0:
    #         self.save_imgs(epoch)
    
    def save_imgs(self, epoch):
        # row,col
        r, c = 5, 5

        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        # rescale [-1, 1] to [0, 1]
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()

if __name__ == "__main__":
    dcgan = DCGAN(28, 28, 1, 5)
    dcgan.train(epochs=100000, batch_size=32, save_interval=1000)

