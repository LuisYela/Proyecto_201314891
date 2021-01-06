import scipy.io

def get_dataFile():
    data = scipy.io.loadmat('datasets/data.mat')
    
    train_X = data['X'].T
    train_Y = data['y'].T
    val_X = data['Xval'].T
    val_Y = data['yval'].T

    #print('data[\'y\']', data['y'])
    #print('data[\'y\']', type(data['y']))
    #print('data[\'y\']', (data['y']).shape)
    #print('data[\'y\']', (data['y'][0]).shape)
    #print('data[\'y\']', type(data['y'][0]))
    #print('data[\'y\'].T', data['y'].T)
    print('data[\'y\'].T', (data['y'].T).shape)
    print('data[\'X\'].T', (data['X'].T).shape)
    #print('data[\'y\'].T', type(data['y'].T))

    return train_X, train_Y, val_X, val_Y
