import pickle


def load_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_to_file(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
