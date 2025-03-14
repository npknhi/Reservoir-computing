
def weight_transfer(trained_model, target_model):
    """ Utility function to transfer weights from trained model to target model """
    weights_dict = {}
    layer_names = []
    # First get dictionary of trained parameters based on name of the layers
    for layer in trained_model.layers:
        weights_dict[layer.name] = layer.get_weights()
        layer_names.append(layer.name)
    
    transfered_layers = 0
    for layer in target_model.layers:
        if layer.name in weights_dict.keys():
            try:
                layer.set_weights(weights_dict[layer.name])
                print(f"Transfered weights of layer {layer.name} from original model to target model.")
                transfered_layers += 1
                layer_names.remove(layer.name)
            except:
                print(f"Layer shape unmatched: {layer.name}")
                print(f'Trained weights shape: {weights_dict[layer.name][0].shape}, Target weights shape: {layer.get_weights()[0].shape}')
    
    if transfered_layers == len(weights_dict.keys()):
        print("Transfered all layers from original model to target model!")
    else:
        print(f"Only transfered {transfered_layers} / {len(weights_dict.keys())} from original model to target model.")
        print(f'Layers not transfered: {layer_names}')
    target_model._name = target_model.name + '_transferred'
    return target_model


