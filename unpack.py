def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict