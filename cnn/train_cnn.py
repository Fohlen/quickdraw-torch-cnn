import pathlib
from itertools import islice

from torchvision import transforms
from torch.utils.data import DataLoader, get_worker_info
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from quickdraw_dataset import QuickdrawDataset
from cnn import CNN


def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    split_size = len(dataset.files) // worker_info.num_workers

    dataset.data = dataset.files[worker_id * split_size:(worker_id + 1) * split_size]

if __name__ == '__main__':
    data_dir = pathlib.Path.cwd() / "data"
    results_dir = pathlib.Path.cwd() / "results"
    bitmaps = list(data_dir.glob("*.npy"))

    dataset = QuickdrawDataset(bitmaps, limit=slice(1000), transform=transforms.ToTensor())
    train_dataloader = DataLoader(dataset, batch_size=100, 
        worker_init_fn=worker_init_fn, num_workers=4
    )

    cnn = CNN(num_classes=len(bitmaps))
    trainer = pl.Trainer(
        max_epochs=2500,
        strategy="dp", 
        accelerator="cpu",
        val_check_interval=100, 
        callbacks=[
        EarlyStopping(monitor="val_loss"),
        ModelCheckpoint(
            monitor='val_loss',
            dirpath=results_dir,
            filename='sample-quickdraw-{epoch:02d}-{val_loss:.2f}.ckpt'
        )
    ])
    trainer.fit(cnn, train_dataloader)
    trainer.save_checkpoint(results_dir / "sample-quickdraw.ckpt")
