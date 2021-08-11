import json
import numpy as np
import os
from collections import defaultdict

import torch


def batch_data(data, batch_size, seed):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(seed)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)


def read_dir(data_dir):
    clients = []
    num_samples = {}
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        for client, samples in zip(cdata['users'], cdata['num_samples']):
            if client in num_samples.keys():
                print("client {} duplicate".format(client))
                samples += num_samples[client]
            num_samples.update({client: samples})
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    num_samples = [num_samples[client] for client in clients]

    return clients, groups, data, num_samples


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data, train_samples = read_dir(train_data_dir)
    test_clients, test_groups, test_data, test_samples = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


def convert_to_tensor(data, clients):
    data_points = []
    targets = []

    for i, client in enumerate(clients):
        samples = data[client]

        data_points.extend([sample for sample in samples['x']]) # torch.Tensor(sample).reshape((28,28))
        targets.extend(samples['y'])

    print(type(data_points))
    print(len(data_points))

    print("num clients: ", len(clients))
    print("num samples: ", len(data_points))
    # print("data sample: ", data_points[0])
    print("len:         ", len(data_points[0]))
    print("target:      ", targets[0])

    targets_t = torch.IntTensor(targets)
    targets = []

    data_t = torch.Tensor(data_points)
    data_points = []
    data_t = data_t.reshape((len(data_t),28,28))

    # print("sample data:   ", data_t[0])
    print("type:          ", type(data_t[0]))
    print("lenx:          ", len(data_t[0]))
    print("leny:          ", len(data_t[0][0]))
    print("sample target: ", targets_t[0])
    print("type:          ", type(targets_t[0]))

    return data_t, targets_t


def read_data_pytorch(data_dir):
    clients, groups, data, samples = read_dir(data_dir)
    data, targets = convert_to_tensor(data, clients)

    print(len(samples))
    print(samples[0:5])

    return data, targets, samples


def read_dataset_pytorch(train_data_dir, test_data_dir):
    train_data, train_targets, train_samples = read_data_pytorch(train_data_dir)
    test_data, test_targets, test_samples = read_data_pytorch(test_data_dir)

    torch.save((train_data, train_targets, train_samples), train_data_dir+"/../training.pt")
    torch.save((test_data, test_targets, test_samples), train_data_dir+"/../test.pt")

    exit()

    return train_data, train_targets, train_samples, test_data, test_targets, test_samples
