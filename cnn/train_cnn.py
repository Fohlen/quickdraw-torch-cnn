import pathlib
import argparse

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


def train_model(
    data_dir: pathlib.Path,
    checkpoints_dir: pathlib.Path,
    num_samples: int,
    num_batch: int,
    num_workers: int,
    max_epochs: int
):
    bitmaps = list(data_dir.glob("*.npy"))

    dataset = QuickdrawDataset(bitmaps, limit=slice(num_samples), transform=transforms.ToTensor())
    train_dataloader = DataLoader(dataset, batch_size=num_batch, 
        worker_init_fn=worker_init_fn, num_workers=num_workers
    )

    cnn = CNN(num_classes=len(bitmaps))
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        val_check_interval=100, 
        callbacks=[
        EarlyStopping(monitor="train_loss"),
        ModelCheckpoint(
            monitor='train_loss',
            dirpath=checkpoints_dir,
            filename='sample-quickdraw-{epoch:02d}-{val_loss:.2f}.ckpt'
        )
    ])
    trainer.fit(cnn, train_dataloader)
    trainer.save_checkpoint("sample-quickdraw.ckpt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir', 
        help="Which directory the data is in", 
        nargs="?", 
        type=pathlib.Path, 
        default=pathlib.Path.cwd() / "data"
    )
    parser.add_argument(
        '--checkpoint-dir', 
        help="Which directory the data is in", 
        nargs="?", 
        type=pathlib.Path, 
        default=pathlib.Path.cwd() / "checkpoints"
    )
    parser.add_argument(
        '--num-samples',
        help="The number of samples that should be considered per epoch",
        nargs="?",
        type=int,
        default=10000
    )
    parser.add_argument(
        '--num-batch',
        help="The number of samples that should be considered per batch",
        nargs="?",
        type=int,
        default=100
    )
    parser.add_argument(
        '--num-workers',
        help="The number of workers used for training",
        nargs="?",
        type=int,
        default=4
    )
    parser.add_argument(
        '--max-epochs',
        help="The maximum number of epochs to run",
        nargs="?",
        type=int,
        default=1000
    )

    args = parser.parse_args()
    train_model(
        args.data_dir,
        args.checkpoint_dir,
        args.num_samples,
        args.num_batch,
        args.num_workers,
        args.max_epochs
    )    
