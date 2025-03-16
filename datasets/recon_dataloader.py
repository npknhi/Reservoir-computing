import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import os

class InpaintingImageDataGen(tf.keras.utils.Sequence):
    def __init__(self, data_array, batch_size, training=True, corruption_ratio=0.25, seed=None):
        super(InpaintingImageDataGen, self).__init__()
        self.data_array = data_array
        self.batch_size = batch_size
        self.training = training
        self.corruption_ratio = corruption_ratio
        self.seed = seed
        self.height, self.width = self.data_array.shape[1], self.data_array.shape[2]
    
    def __len__(self):
        return len(self.data_array) // self.batch_size
    
    def __getitem__(self, idx):
        left_bound = idx * self.batch_size
        right_bound = (idx + 1) * self.batch_size
        # Handle the case where right bound overpasses the number of samples
        if right_bound > len(self.data_array):
            right_bound = len(self.data_array)
            # Shift the left bound correspondingly to the left side
            left_bound = right_bound - self.batch_size
        
        item_idx = 0
        x_batch, y_batch = np.zeros((self.batch_size, self.height, self.width, 3)), np.zeros((self.batch_size, self.height, self.width, 3))
        
        for current_idx in range(left_bound, right_bound):
            img = self.data_array[current_idx].astype('float32')/255.
            # First just randomly flip the image to the left and the right
            if np.random.normal() > 0 and training:
                img = np.flip(img, axis=1)

            if self.training:
                self.corruption_ratio = np.random.uniform(0.25, 0.75)
            mask = np.random.choice([0., 1.], size=(self.height, self.width, 1), replace=True, p=[self.corruption_ratio, 1. - self.corruption_ratio]).astype('float32')

            img_corrupted = img * mask
            # Assign to batch
            x_batch[item_idx] = img_corrupted
            y_batch[item_idx] = img
            item_idx += 1
        
        return x_batch, y_batch
    
    def on_epoch_end(self):
        # Shuffle the data array
        if self.seed is not None:
            np.random.seed(self.seed)
            np.random.shuffle(self.data_array)

def recon_loader(dataset):
    if dataset == 'stl10':
        data = np.load(f'datasets/stl10.npz')  
        test_path = 'datasets/stl10_test.npz'
        X_train, X_test = data['X_train'], data['X_test']
    
    train_ds = InpaintingImageDataGen(X_train, batch_size=50)
    
    if not os.path.exists(test_path):
        X_test, y_test = InpaintingImageDataGen(X_test, batch_size=8000)[0]
        test_ds = (X_test, y_test)
        np.savez(test_path, X_test=X_test, y_test=y_test)
    else:
        test_ds = np.load(test_path)
        test_ds = (test_ds['X_test'], test_ds['y_test'])
    return train_ds, test_ds
