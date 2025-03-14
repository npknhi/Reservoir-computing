import os
import urllib.request
import tarfile
import numpy as np

def download_and_extract(url, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    tar_path = os.path.join(dest_folder, 'stl10_binary.tar.gz')
    
    if not os.path.exists(tar_path):
        print('Downloading STL-10 dataset...')
        urllib.request.urlretrieve(url, tar_path)
        print('Download complete.')
    else:
        print('Tar file already exists, skipping download.')
    
    print('Extracting dataset...')
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=dest_folder)
    print('Extraction complete.')

def load_stl10_images(filename, num_images):
    with open(filename, 'rb') as f:
        images = np.fromfile(f, dtype=np.uint8)
        images = images.reshape(num_images, 3, 96, 96) 
        images = images.transpose(0, 3, 2, 1)  # Convert to (N, H, W, C)
    return images

def load_stl10_labels(filename):
    with open(filename, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

def save_to_npz(data_folder, output_file):
    X_train = load_stl10_images(os.path.join(data_folder, 'train_X.bin'), 5000)
    y_train = load_stl10_labels(os.path.join(data_folder, 'train_y.bin'))
    X_test = load_stl10_images(os.path.join(data_folder, 'test_X.bin'), 8000)
    y_test = load_stl10_labels(os.path.join(data_folder, 'test_y.bin'))
    
    with open(os.path.join(data_folder, 'class_names.txt'), 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    np.savez(output_file,
             X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test,
             class_names=class_names)
    print(f"Dataset saved as {output_file}")

if __name__ == "__main__":
    Stl10_URL = 'https://cs.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
    raw_data_folder = './datasets'
    extracted_data_folder = './datasets/stl10_binary'
    output_file = './datasets/stl10.npz'
    
    download_and_extract(Stl10_URL, raw_data_folder)
    save_to_npz(extracted_data_folder, output_file)
