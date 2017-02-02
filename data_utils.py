import numpy as np

def load(filename, test=False):
    f = open(filename, 'rb')
    data = np.loadtxt(f, dtype='int', delimiter=',', skiprows=1)
    f.close()
    
    if test:
        # ignore first column of ids
        X = data[:, 1:]
        return X
    else:
        X,y = data[:, 1:-1], data[:, -1]
        return X,y


def load_train(filename):
    X, y = load(filename, False)
    return X, y


def load_test(filename):
    X = load(filename, True)
    return X


def write_test(filename, vals):
    f = open(filename, 'w')
    
    f.write("Id,PES1\n")
    for i in xrange(len(vals)):
        f.write('%s,%d\n' %(str(i), int(vals[i])))
    
    f.close()


if __name__=='__main__':
    train = './data/train_2008.csv'
    
    X, y = load_train(train)
    print 'X: ', X.shape
    print 'y: ', y.shape
    
