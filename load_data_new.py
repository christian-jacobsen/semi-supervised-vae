import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader, TensorDataset


def load_data_new(data_dir, batch_size, shuff = True):
    """Return data loader

    Args:
        data_dir: directory to hdf5 file, e.g. `dir/to/kle4225_lhs256.hdf5`
        batch_size (int): mini-batch size for loading data

    Returns:
        (data_loader (torch.utils.data.DataLoader), stats)
    """

    with h5py.File(data_dir, 'r') as f:
        #x_data = f['input'][()]
        z_data = f['generative_params'][()]
        x_data = f['permeability'][()]
        y_data = f['output'][()]
        
    
    np.random.seed(201)
    nd = np.shape(z_data)[0]
    perm = np.random.permutation(nd)
    z_data = z_data[perm,:]
    x_data = x_data[perm,:,:,:]
    y_data = y_data[perm,:,:,:]
    print("permeability data shape: {}".format(x_data.shape))
    print("generative factor data shape: {}".format(z_data.shape))
    print("output data shape: {}".format(y_data.shape))
    
    

    kwargs = {'num_workers': 0,
              'pin_memory': True} if torch.cuda.is_available() else {}

    dataset = TensorDataset(torch.tensor(z_data), torch.tensor(x_data), torch.tensor(y_data))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuff, **kwargs)

    # simple statistics of output data
    y_data_mean = np.mean(y_data, 0)
    y_data_var = np.sum((y_data - y_data_mean) ** 2)
    stats = {}
    stats['y_mean'] = y_data_mean
    stats['y_var'] = y_data_var

    return data_loader, stats
