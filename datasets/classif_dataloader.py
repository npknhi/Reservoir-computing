import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import os

def augment_fn(X, y, image_width, augmentation=False):
    if augmentation:
        # Pad image with 4 pixels on each side
        X = tf.image.resize_with_crop_or_pad(X, int(image_width*1.25), int(image_width*1.25))

        # Randomly crop back to 32x32
        X = tf.image.random_crop(X, size=[image_width, image_width, 3])

        # Apply random horizontal flip
        X = tf.image.random_flip_left_right(X)

        # Apply random cutout
        X = tf.expand_dims(X, axis=0)  # Add batch dimension
        X = tfa.image.random_cutout(X, mask_size=(int(image_width*0.1)*2, int(image_width*0.1)*2))
        X = tf.squeeze(X, axis=0)
    return X, y

def dataset_generator(inputs, labels, image_width, batch_size, augmentation):
    ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
    ds = ds.map(lambda X, y: augment_fn(X, y, image_width, augmentation=augmentation), 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(len(inputs)).batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

def classif_loader(dataset, augmentation=True):
    if dataset == 'stl10':
        data = np.load(f'datasets/stl10.npz')  
        X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
        image_width = 96

    # Normalize the images to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    # Convert labels from 1-10 to 0-9
    y_train -= 1
    y_test -= 1

    # Ensure labels are one-hot
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    # Use the dataset generator to create the training, testing datasets
    train_ds = dataset_generator(X_train, y_train, image_width, batch_size=50, augmentation=augmentation)
    test_ds = dataset_generator(X_test, y_test, image_width, batch_size=50, augmentation=False)

    return train_ds, test_ds

def image_visualize(dataset):
    for images, images_ag in dataset.take(1):  # Take one batch
        for image in images:
            image = image.numpy()   
            # Plot the image
            plt.imshow(image)
            plt.axis("off") 
            plt.show()

if __name__ == "__main__":
    train_ds, test_ds = classif_loader('stl10')
    image_visualize(train_ds)
    


