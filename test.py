import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

model = load_model('cancer_detector.keras')

test_dir = 'test'

print(f"Contents of {test_dir}:")
print(os.listdir(test_dir)[:10])

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(96, 96),
    batch_size=32,
    class_mode=None,
    shuffle=False,
    classes=['.']
)

image_ids = [os.path.basename(f) for f in test_generator.filenames]
image_ids = [f[:-4] if f.endswith('.tif') else f for f in image_ids]

predictions = model.predict(test_generator)

submission_df = pd.DataFrame({
    'id': image_ids,
    'label': predictions.flatten()
})

submission_df.to_csv('submission.csv', index=False)

print("Submission file saved as submission.csv")
