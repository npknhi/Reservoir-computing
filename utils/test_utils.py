import tensorflow as tf
from utils.train_utils import ssim, psnr

def test(test_ds, model):
    model_path = f'checkpoints/model/{model.name}.h5'
    model.load_weights(model_path)
    print("Successfully loaded the trained model!")

    model_info = model.name
    task = model_info.split('_')[0]

    if task == 'classif':       
        model.compile(metrics=['accuracy'], loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))   
        loss, accuracy = model.evaluate(test_ds)
        print(f'{model_info} - acc: {accuracy} - loss: {loss}')

    elif task == 'recon':
        model.compile(metrics=[ssim, psnr], loss=tf.keras.losses.MeanAbsoluteError())
        loss, SSIM, PSNR = model.evaluate(test_ds[0], test_ds[1], batch_size=50)  
        print(f'{model_info} - ssim: {SSIM} - psnr: {PSNR} - loss: {loss}')

    return 
