import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

df = pd.read_csv('train_labels.csv')

df['label'] = df['label'].astype(str)
df['id'] = df['id'] + '.tif'

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    directory='train/',
    x_col='id',
    y_col='label',
    target_size=(96, 96),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

base_model = ResNet50(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=10) 

test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'test/', 
    target_size=(96, 96), 
    batch_size=32, 
    class_mode=None,
    shuffle=False
)
predictions = model.predict(test_generator)