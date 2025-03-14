import tensorflow as tf
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping # type: ignore

def ssim(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, tf.clip_by_value(y_pred, 0., 1.), max_val=1.0))

def psnr(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, tf.clip_by_value(y_pred, 0., 1.) + 1e-3, max_val=1.0))

def train(train_ds, test_ds, model, epochs, lr=1e-3):
    task = model.name.split('_')[0]
    # Optimizer
    total_steps = len(train_ds) * epochs
    cosine_lr_decay = tf.keras.optimizers.schedules.CosineDecay(lr, total_steps, alpha=1e-5)
    opt = Adam(learning_rate=cosine_lr_decay)
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
    csv_logger = CSVLogger(f'checkpoints/history/{model.name}.csv', append=True)  
    callbacks = [csv_logger, early_stopping]

    # Compile and train the model
    if task == 'classif':
        model.compile(optimizer=opt, metrics=['accuracy'], loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))    
        history = model.fit(train_ds, steps_per_epoch=len(train_ds), epochs=epochs, 
                            validation_data=test_ds, validation_steps=len(test_ds), verbose=1, callbacks=callbacks)
    elif task == 'recon':
        model.compile(optimizer=opt, metrics=[ssim, psnr], loss=tf.keras.losses.MeanAbsoluteError())
        history = model.fit(train_ds, steps_per_epoch=len(train_ds), epochs=epochs, 
                            validation_data=test_ds, validation_batch_size=50, verbose=1, callbacks=callbacks)
    
    print(history)
    # Save the model
    tf.keras.models.save_model(model, f'checkpoints/model/{model.name}.h5')
    return
