import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

model = load_model('cancer_detector_1_epoch.keras')

df_test = pd.read_csv('test_labels.csv')
df_test['id'] = df_test['id'].astype(str) + '.tif'

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory='test/',
    x_col='id',
    y_col=None,
    target_size=(96, 96),
    batch_size=32,
    class_mode=None,
    shuffle=False
)

predictions = model.predict(test_generator)

print(predictions)