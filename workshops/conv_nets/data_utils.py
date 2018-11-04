import numpy as np
import os
import pickle

def get_cifar10_path():
    """
    Location where downloaded CIFAR-10 dataset should be stored.
    """
    return os.path.join(os.path.dirname(__file__), "data", "cifar10.pkl")

def download_cifar10():
    """
    Downloads CIFAR-10 from the web to a predefined local storage location.
    """
    cifar10_url = r"https://onedrive.live.com/download?cid=575F408FB4AD0B98&resid=575F408FB4AD0B98%21170873&authkey=AAITnCQTwq0cMmg"
    cifar10_path = get_cifar10_path()
    cifar10_dir_path = os.path.dirname(cifar10_path)

    if os.path.exists(cifar10_dir_path):
        print("Found folder %s, assuming CIFAR-10 is present." % cifar10_dir_path)
    else:
        import wget
        os.makedirs(cifar10_dir_path)
        print("Downloading CIFAR-10 to %s..." % cifar10_path)
        wget.download(url=cifar10_url, out=cifar10_path)
        print("\nDone.")

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f, encoding="latin1")

def load_cifar10():
    cifar10_path = get_cifar10_path()

    cifar10_dict = load_pickle(cifar10_path)

    class_names, images_dev, labels_dev, images_test, labels_test = \
        [cifar10_dict[k] for k in ["class_names", "images_dev", "labels_dev", "images_test", "labels_test"]]

    class_labels = np.unique(np.concatenate([labels_dev, labels_test]))

    print("-" * 70)
    print("Loaded CIFAR-10.")
    print("-" * 70)
    print("Dev data shape: %s." % str(images_dev.shape))
    print("Dev labels shape: %s." % str(labels_dev.shape))
    print("Test data shape: %s." % str(images_test.shape))
    print("Test labels shape: %s." % str(labels_test.shape))
    print("Class labels: %s." % str(class_labels))
    print("Class names: %s." % str(class_names))

    return class_names, images_dev, labels_dev, images_test, labels_test

def load_cifar10_batches(file_paths):
    dicts = [load_pickle(fn) for fn in file_paths]
    x = np.concatenate([d["data"] for d in dicts]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    y = np.concatenate([np.asarray(d["labels"]) for d in dicts])
    return x, y

def load_cifar10_original(root):
    """
    Loads CIFAR-10 dataset from original distribution.
    """
    num_dev_batches = 5
    original_batch_file_paths = [os.path.join(root, "data_batch_%d" % b) for b in range(1, 1 + num_dev_batches)]
    images_dev, labels_dev = load_cifar10_batches(original_batch_file_paths)
    images_test, labels_test = load_cifar10_batches([os.path.join(root, "test_batch"),])
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    return class_names, images_dev, labels_dev, images_test, labels_test

def train_val_split(images_dev, labels_dev, val_fraction):
    num_dev = images_dev.shape[0]
    num_val = int(num_dev * val_fraction)
    num_train = num_dev - num_val

    perm = np.random.permutation(num_dev)
    indices_train = perm[:num_train]
    indices_val = perm[num_train:]

    images_train = images_dev[indices_train]
    labels_train = labels_dev[indices_train]
    images_val = images_dev[indices_val]
    labels_val = labels_dev[indices_val]

    return images_train, labels_train, images_val, labels_val

def save_cifar10(class_names, images_dev, labels_dev, images_test, labels_test, file_path):
    cifar10_dict = {
        "images_dev" : images_dev,
        "labels_dev" : labels_dev,
        "images_test" : images_test,
        "labels_test" : labels_test,
        "class_names" : class_names,
        }
    with open(file_path, "wb") as f:
        pickle.dump(cifar10_dict, f)
