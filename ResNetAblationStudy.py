
# 0) 라이브러리 불러오기
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

import numpy as np
import matplotlib.pyplot as plt


# 1) ResNet 기본 블록 구성하기
# 기본 블록 구성하기
def conv_block(input_layer,
               channel,
               kernel_size,
               strides=1,
               activation='relu'):
    x = keras.layers.Conv2D(filters=channel,
                            kernel_size=kernel_size,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(1e-4),
                            padding='same',
                            strides=strides
                            )(input_layer)

    x = keras.layers.BatchNormalization()(x)

    if activation:
        x = keras.layers.Activation(activation)(x)

    return x


# ResNet 기본 블록 구성하기
def build_resnet_block(input_layer,
                       num_cnn=3,
                       channel=64,
                       block_num=0,
                       is_50=True):
    x = input_layer

    # ResNet-50
    if is_50:
        for i in range(num_cnn):
            if i == 0:
                shortcut = conv_block(x, channel * 4, (1, 1), strides=2, activation=None)
                x = conv_block(x, channel, (1, 1), strides=2)
                x = conv_block(x, channel, (3, 3))
                x = conv_block(x, channel * 4, (1, 1), activation=None)
            else:
                shortcut = x
                x = conv_block(x, channel, (1, 1))
                x = conv_block(x, channel, (3, 3))
                x = conv_block(x, channel * 4, (1, 1), activation=None)

            x = keras.layers.Add()([x, shortcut])
            x = keras.layers.Activation('relu')(x)

    # ResNet-34
    else:
        for i in range(num_cnn):
            if block_num > 0 and i == 0:
                shortcut = conv_block(x, channel, (1, 1), strides=2, activation=None)
                x = conv_block(x, channel, (3, 3), strides=2)
                x = conv_block(x, channel, (3, 3), activation=None)
            else:
                shortcut = x
                x = conv_block(x, channel, (3, 3))
                x = conv_block(x, channel, (3, 3), activation=None)

            x = keras.layers.Add()([x, shortcut])
            x = keras.layers.Activation('relu')(x)

    return x


# ResNet 모델 구성하기
def build_resnet(input_shape=(32, 32, 3),
                 num_cnn_list=[3, 4, 6, 3],
                 channel_list=[64, 128, 256, 512],
                 num_classes=10,
                 is_50=True,
                 activation='softmax',
                 name='ResNet_50'):
    assert len(num_cnn_list) == len(channel_list)  # 모델을 만들기 전에 config list들이 같은 길이인지 확인합니다.

    input_layer = keras.layers.Input(shape=input_shape)  # input layer를 만들어둡니다.

    # first layer
    x = conv_block(input_layer, 64, (7, 7), strides=2)
    x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)

    # config list들의 길이만큼 반복해서 Residual block 블록을 생성합니다.
    for block_num, (num_cnn, channel) in enumerate(zip(num_cnn_list, channel_list)):
        x = build_resnet_block(x,
                               num_cnn=num_cnn,
                               channel=channel,
                               block_num=block_num,
                               is_50=is_50)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(num_classes,
                           activation=activation,
                           kernel_initializer='he_normal'
                           )(x)

    model = keras.Model(inputs=input_layer, outputs=x, name=name)

    return model

# 2) ResNet-34, ResNet-50 Complete Model
resnet_34 = build_resnet(input_shape=(32, 32,3), is_50=False)
resnet_34.summary()

resnet_50 = build_resnet(input_shape=(32, 32,3), is_50=True)
resnet_50.summary()

# 3) 일반 네트워크(plain network) 만들기
# PlainNet 기본 블록 구성하기
def build_plain_block(input_layer,
                      num_cnn=3,
                      channel=64,
                      block_num=0,
                      is_50=True):
    x = input_layer

    # plain-50
    if is_50:
        for i in range(num_cnn):
            if block_num > 0 and i == 0:
                x = conv_block(x, channel, (1, 1), strides=2)
                x = conv_block(x, channel, (3, 3))
                x = conv_block(x, channel * 4, (1, 1))
            else:
                x = conv_block(x, channel, (1, 1))
                x = conv_block(x, channel, (3, 3))
                x = conv_block(x, channel * 4, (1, 1))

    # plain-34
    else:
        for i in range(num_cnn):
            if block_num > 0 and i == 0:
                x = conv_block(x, channel, (3, 3), strides=2)
                x = conv_block(x, channel, (3, 3))
            else:
                x = conv_block(x, channel, (3, 3))
                x = conv_block(x, channel, (3, 3))

    return x


# PlainNet 모델 구성하기
def build_plainnet(input_shape=(32, 32, 3),
                   num_cnn_list=[3, 4, 6, 3],
                   channel_list=[64, 128, 256, 512],
                   num_classes=10,
                   is_50=True,
                   activation='softmax',
                   name='Plain_50'):
    assert len(num_cnn_list) == len(channel_list)

    input_layer = keras.layers.Input(shape=input_shape)

    # first layer
    x = conv_block(input_layer, 64, (7, 7), strides=2)
    x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)

    # config list들의 길이만큼 반복해서 plain block 블록을 생성합니다.
    for block_num, (num_cnn, channel) in enumerate(zip(num_cnn_list, channel_list)):
        x = build_plain_block(x,
                              num_cnn=num_cnn,
                              channel=channel,
                              block_num=block_num,
                              is_50=is_50)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(num_classes,
                           activation=activation,
                           kernel_initializer='he_normal'
                           )(x)

    model = keras.Model(inputs=input_layer, outputs=x, name=name)

    return model


# 4) ResNet-50 vs Plain-50 또는 ResNet-34 vs Plain-34
import urllib3
urllib3.disable_warnings()

# cats_vs_dogs 데이터셋이 tensorflow API 오류로 불러오지 못하는 경우가 있을 경우
# 로드하기 전에 이것을 추가해서 새 URL을 설정할 수 있습니다.
setattr(tfds.image_classification.cats_vs_dogs, '_URL',"https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip")

(ds_train, ds_test), ds_info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    shuffle_files=True,
    as_supervised=True, # 데이터가 튜플로 반환, False는 dictionarty형태로 반환
    with_info=True, # 데이터의 정보를 같이 반환한다.
)

# 데이터 전처리하기(정규화 함수 만들기)
def normalize_and_resize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.image.resize(image, (224,224))
    image = tf.cast(image, tf.float32) / 255.
    return image, label

def apply_normalize_on_dataset(ds, is_test=False, batch_size=16):
    ds = ds.map(
        normalize_and_resize_img,
        num_parallel_calls=1
    )
    ds = ds.batch(batch_size)
    if not is_test:
        ds = ds.repeat()
        ds = ds.shuffle(200)
    #ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

BATCH_SIZE = 32
EPOCH = 10
#BATCH_SIZE = 3
#EPOCH = 1

ds_train = apply_normalize_on_dataset(ds_train, batch_size=BATCH_SIZE)
ds_test = apply_normalize_on_dataset(ds_test, batch_size=BATCH_SIZE)

############################################# resnet
# resnet_34 학습하기
resnet_34 = build_resnet(input_shape=(224, 224, 3),
                         num_classes=1,
                         is_50=False,
                         activation='sigmoid',
                         name='ResNet_34')

resnet_34.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy'],
)

history_resnet_34 = resnet_34.fit(
    ds_train,
    steps_per_epoch=int(ds_info.splits['train[:80%]'].num_examples/BATCH_SIZE),
    validation_steps=int(ds_info.splits['train[80%:]'].num_examples/BATCH_SIZE),
    epochs=EPOCH,
    validation_data=ds_test,
    verbose=1,
    use_multiprocessing=True,
)

# resnet_50 학습하기
resnet_50 = build_resnet(input_shape=(224, 224, 3),
                         num_classes=1,
                         is_50=True,
                         activation='sigmoid')

resnet_50.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy'],
)

history_resnet_50 = resnet_50.fit(
    ds_train,
    steps_per_epoch=int(ds_info.splits['train[:80%]'].num_examples/BATCH_SIZE),
    validation_steps=int(ds_info.splits['train[80%:]'].num_examples/BATCH_SIZE),
    epochs=EPOCH,
    validation_data=ds_test,
    verbose=1,
    use_multiprocessing=True,
)

############################################# plainnet
# plain_34 학습하기
plain_34 = build_plainnet(input_shape=(224, 224, 3),
                          num_classes=1,
                          is_50=False,
                          activation='sigmoid',
                          name='Plain_34')

plain_34.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy'],
)

history_plain_34 = plain_34.fit(
    ds_train,
    steps_per_epoch=int(ds_info.splits['train[:80%]'].num_examples/BATCH_SIZE),
    validation_steps=int(ds_info.splits['train[80%:]'].num_examples/BATCH_SIZE),
    epochs=EPOCH,
    validation_data=ds_test,
    verbose=1,
    use_multiprocessing=True,
)

# plain_50 학습하기
plain_50 = build_plainnet(input_shape=(224, 224, 3),
                          num_classes=1,
                          is_50=True,
                          activation='sigmoid')

plain_50.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy'],
)

history_plain_50 = plain_50.fit(
    ds_train,
    steps_per_epoch=int(ds_info.splits['train[:80%]'].num_examples/BATCH_SIZE),
    validation_steps=int(ds_info.splits['train[80%:]'].num_examples/BATCH_SIZE),
    epochs=EPOCH,
    validation_data=ds_test,
    verbose=1,
    use_multiprocessing=True,
)

# 시각화
# 시각화하여 결과 비교하기
plt.figure(figsize=(10, 20))
plt.subplot(2, 1, 1)

plt.plot(history_resnet_34.history['loss'], 'r')
plt.plot(history_resnet_50.history['loss'], 'b')
plt.plot(history_plain_34.history['loss'], 'y', linestyle='dashed')
plt.plot(history_plain_50.history['loss'], 'g', linestyle='dashed')
plt.title('Model training loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['ResNet_34','ResNet_50','ResNet_34_Plain','ResNet_50_Plain'], loc='upper left')

plt.subplot(2, 1, 2)
plt.plot(history_resnet_34.history['val_accuracy'], 'r')
plt.plot(history_resnet_50.history['val_accuracy'], 'b')
plt.plot(history_plain_34.history['val_accuracy'], 'y', linestyle='dashed')
plt.plot(history_plain_50.history['val_accuracy'], 'g', linestyle='dashed')
plt.title('Model validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['ResNet_34','ResNet_50','ResNet_34_Plain','ResNet_50_Plain'], loc='upper left')
plt.show()