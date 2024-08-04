import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

df = pd.read_csv('train_labels.csv')

df['label'] = df['label'].astype(str)
df['id'] = df['id'] + '.tif'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    directory='train/',
    x_col='id',
    y_col='label',
    target_size=(96, 96),
    batch_size=32,
    class_mode='binary',
    shuffle=True,
    subset='training'
)

validation_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    directory='train/',
    x_col='id',
    y_col='label',
    target_size=(96, 96),
    batch_size=32,
    class_mode='binary',
    shuffle=True,
    subset='validation'
)

input_tensor = Input(shape=(96, 96, 3))

base_model = ResNet50(input_tensor=input_tensor, include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
output_tensor = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output_tensor)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    ModelCheckpoint('best_model.keras', save_best_only=True),
    EarlyStopping(patience=10, restore_best_weights=True)
]

model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=callbacks
)

model.save('cancer_detector.keras')
