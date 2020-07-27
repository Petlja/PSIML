from torch.utils.data import DataLoader

def generate_batches(dataset, batch_size, shuffle=False, drop_last=True, device="cpu"):
    """
    A generator function that wraps the PyTorch DataLoader. It will ensure that ech tensor is on the right device location
    
    Args:
        dataset (TwitterDataset): instance of the PyTorch Dataset, that should be divided into batches
        batch_size (int): size of the batch
        shuffle (bool): a flag whether the dataset should be shuffled
        drop_last (bool): a flag whether the last batch should be dropped if the dataset size is not divideable by the batch size
        device (str): string denoting the device, the values are: "cpu" or "gpu"
    """
    # initialize the PyTorch DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )

    # iterate through all batches in the dataset
    for data_dict in dataloader:
        out_data_dict = {}
        # send the tensor to the appropriate device
        for name, tensor in data_dict.items():
            out_data_dict[name] = tensor.to(device)

        yield out_data_dict