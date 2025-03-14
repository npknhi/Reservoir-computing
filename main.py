from datasets.classif_dataloader import classif_loader
from datasets.recon_dataloader import recon_loader
from models.classification_models import classif_model
from models.recontruction_models import reconstruction_model
from utils.transfer_weights import weight_transfer
from utils.train_utils import train
from utils.test_utils import test

import numpy as np

def main():     
    epoch = 100

    # Classif model  
    train_ds, test_ds = classif_loader(dataset='stl10')
    for config in [1,2,3]:
        model = classif_model(config, trainable=True)                
        train(train_ds, test_ds, model, epoch)
        test(test_ds, model)  

    # Recon model
    train_ds, test_ds = recon_loader(dataset='stl10')
    model = reconstruction_model(config=3, trainable=True)
    train(train_ds, test_ds, model, epoch)
    test(test_ds, model)

    # Classif weight transferred model
    train_ds, test_ds = classif_loader(dataset='stl10')
    recon = reconstruction_model(config=3, trainable=True)
    recon.load_weights(f'checkpoints/model/{recon.name}.h5')
    classif = classif_model(config=3, trainable=False)
    new_classif = weight_transfer(recon, classif)
    train(train_ds, test_ds, new_classif, epoch)
    test(test_ds, new_classif)

    # Recon weight transferred model
    train_ds, test_ds = recon_loader(dataset='stl10')
    classif = classif_model(config=3, trainable=True)
    classif.load_weights(f'checkpoints/model/{classif.name}.h5')
    recon = reconstruction_model(config=3, trainable=False)
    new_recon = weight_transfer(classif, recon)
    train(train_ds, test_ds, new_recon, epoch)
    test(test_ds, new_recon)
    return

if __name__ == '__main__':
    main()
