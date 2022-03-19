import pathlib
from itertools import chain, cycle

import numpy as np
from torch.utils.data import IterableDataset


class QuickdrawDataset(IterableDataset):
    """
    Custom Dataset for loading Quickdraw images.
    """

    def __init__(self, files: list[pathlib.Path], limit: str = slice(None), transform = None):
        super(QuickdrawDataset).__init__()        
        self.files = files
        self.categories = {file.stem: index for index, file in enumerate(files)}
        self.transform = transform
        self.limit = limit

    def process_data(self, file: pathlib.Path) -> np.array:
        with file.open("rb") as stream:
            array = np.load(stream)
            num_samples = array.shape[0]
            for row in array[self.limit, :]:
                bitmap = row.reshape((28, 28))
                image = self.transform(bitmap) if self.transform is not None else bitmap
                yield image, self.categories[file.stem]

    def get_stream(self, files):
        return chain.from_iterable(
            map(self.process_data, files)
        )

    def __iter__(self):
        return self.get_stream(self.files)
