from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def get_generators(base_dir, img_size=(224,224), batch_size=32):
    train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(base_dir,'train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(base_dir,'val'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    test_gen = test_datagen.flow_from_directory(
        os.path.join(base_dir,'test'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen, test_gen
