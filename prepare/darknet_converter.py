import h5py
import numpy as np
from tensorflow.python.keras import Model, Input
import  tensorflow as tf
from backbones.darknet import darknet_body, DarkNetDet


def load_attributes_from_hdf5_group(group, name):
    """Loads attributes of the specified name from the HDF5 group.

    This method deals with an inherent problem
    of HDF5 file which is not able to store
    data larger than HDF5_OBJECT_HEADER_LIMIT bytes.

    # Arguments
        group: A pointer to a HDF5 group.
        name: A name of the attributes to load.

    # Returns
        data: Attributes data.
    """
    if name in group.attrs:
        data = [n.decode('utf8') for n in group.attrs[name]]
    else:
        data = []
        chunk_id = 0
        while ('%s%d' % (name, chunk_id)) in group.attrs:
            data.extend([n.decode('utf8')
                         for n in group.attrs['%s%d' % (name, chunk_id)]])
            chunk_id += 1
    return data


def prepare_darknet_params():
    inputs = Input(shape=(None, None, 3))
    model = Model(inputs, darknet_body(inputs))

    # det_body = DarkNetDet((None, None), 85)
    # model = det_body.get_detection_body()

    with h5py.File('model_data/yolo3_weights.h5', 'r') as f:
        for idx, layer in enumerate(model.layers):
            print(idx, layer.name)
            if len(layer.weights) != 0:
                splited = layer.name.split("_")
                if splited[-1].isdigit():
                    splited[-1] = str(int(splited[-1]) + 1)
                else:
                    splited.append("1")
                h5_name = "_".join(splited)
                weight_names = load_attributes_from_hdf5_group(f[h5_name], 'weight_names')

                weight_values = [np.asarray(f[h5_name][weight_name]) for weight_name in weight_names]
                layer.set_weights(weight_values)

    model.save_weights("yolo_pretrained.h5")




# with h5py.File('model_data/yolo3_weights.h5', 'r') as f:
#     g = f['add_1']
#
#     weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
#     print(weight_names)
#     weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]
#     print(weight_values[0].shape)

if __name__ == '__main__':
    read_darknet_params()
