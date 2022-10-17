import numpy as np
import scipy
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.utils.data
from PIL import Image
import datetime
import copy
import time
import typing
import json

import tqdm
tqdm.tqdm.pandas()

from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
# from IPython import display
import glob
import os

# %matplotlib ipympl
import wandb

from datasets import Dataset, ClassLabel, DatasetDict
from datasets import load_metric
import transformers
from transformers import DefaultDataCollator
from transformers import TrainingArguments, Trainer

from transformers import AutoFeatureExtractor, AutoModelForImageClassification

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor
)
from torchvision import datasets, models, transforms
import kornia

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from sklearn.model_selection import KFold, train_test_split, StratifiedKFold

import wakepy



def scan_sorted_data(path):
    records = []
    for f in glob.glob(path + "/**/*.mat", recursive=True):
        record = dict()
        fname = os.path.basename(f)
        dir_ = os.path.dirname(f)
        relpath = os.path.normpath(os.path.relpath(dir_, path))
        
        record['fname'] = fname
        record['relpath'] = relpath
        record['full_path'] = os.path.join(relpath, fname)
        record['contrast'], record['repetition'] = relpath.split(os.path.sep)
        record['rural'] = record['repetition'][0] == "R"
        record['repetition_idx'] = int(record['repetition'][1])
        record['after'] = record['repetition'].endswith("b")
        
        sects = fname.split("_")
        record['bat_id'] = int(sects[0][2:])
        
        records.append(record)
    
    return pd.DataFrame(records)


def load_images(meta, data_path):
    
    def gm_to_img(arr):
        # return arr
        # image = arr * (255 / 2**15)
        # assert image.max() <= 255, image.max()
        if arr.max() == 0:
            image = np.array(arr).astype("uint16")
            img = Image.fromarray(image)
            return img
        # assert arr.max() > 0, arr.max()
        # b = arr
        if 0 < arr.max() <= 1:
            arr = arr * 2**15
        if arr.max() < 1:
            image = np.array(arr).astype("uint16")
            img = Image.fromarray(image)
            return img
        # if 0 < arr.max() <= 1:
        #     arr = arr * 2**16
        # image = arr * 255
        # image = image.astype('uint8')
        image = np.array(arr).astype("uint16")
        assert image.max() > 0, (image.max(), arr.max())
        img = Image.fromarray(image)
        img = img.rotate(90)
        return img
    def flatten_3d_image(row):
        mat = scipy.io.loadmat(os.path.join(data_path, row['full_path'].iloc[0]))
        volume = mat.get('Data', mat.get('img'))
        # start_range = volume.shape[2] // 5
        # end_range = 4 * (volume.shape[2] // 5)
        start_range = 0
        end_range = volume.shape[2]
        return pd.DataFrame({"image": [gm_to_img(volume[:, :, i]) for i in range(start_range, end_range)]})
    
    df = pd.merge(meta, meta.query("contrast == 'GM'").groupby('fname').apply(flatten_3d_image).reset_index().rename(columns={"level_1":"image_index"}), on="fname")
    
    return df


def collate_fn(examples):
    # pixel_values = torch.stack([example["pixel_values"] for example in examples])
    # labels = torch.tensor([example["label"] for example in examples])
    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


class MyTrainer(Trainer):
    
    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if transformers.trainer.is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = torch.utils.data.IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                persistent_workers=True
            )

        train_sampler = self._get_train_sampler()

        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=transformers.trainer.seed_worker,
            persistent_workers=True
        )
    
    
    def get_eval_dataloader(self, eval_dataset: typing.Optional[Dataset] = None) -> torch.utils.data.DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not accepted by
                the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        if transformers.trainer.is_datasets_available() and isinstance(eval_dataset, Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return torch.utils.data.DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=True
        )



def run_train(train_ds, val_ds, model, feature_extractor, output_dir):
    
    # Run training
    batch_size = 16 # batch size for training and evaluation
    
    metric = load_metric("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # def transforms(examples):
    #     examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    #     del examples["image"]
    #     return examples


    training_args = TrainingArguments(
        output_dir=output_dir, 
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy = "epoch",
        remove_unused_columns=False,
        report_to="wandb",  # enable logging to W&B,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        load_best_model_at_end=True,
        # metric_for_best_model="accuracy",
        num_train_epochs=5,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        # logging_steps=50,
    )
    trainer = MyTrainer(
        model,
        training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    train_results = trainer.train()
    wandb.finish()

    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    
    return model


def predict_one(image, model, val_transforms):
    encoding = torch.stack((val_transforms(image), )).to("cuda:0")
    with torch.no_grad():
        outputs = model(encoding)
        logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return predicted_class_idx


def run_test(test_df, model, val_transforms):
    df2 = test_df.copy()
    df2['pred'] = df2['image'].progress_apply(predict_one, model=model, val_transforms=val_transforms)
    df2['correct'] = df2['pred'] == df2['rural'].astype(int)

    # process_grad_cam(model, df2, val_transforms)
    
    def grp_pred(grp):
        return np.round(grp.mean())
    test_pred: pd.DataFrame() = df2.groupby(["bat_id", "repetition"])[['rural', 'pred']].agg({"rural": 'mean', "pred": grp_pred})
    test_accuracy = (test_pred["rural"] == test_pred['pred']).mean()
    print(test_accuracy, test_pred["rural"].sum())
    print(test_pred.to_markdown())


def explain(val_ds, val_transforms, model):
    from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    idx = 10
    image = val_ds[idx]['image']
    img = image.point(lambda p: p * ((2 ** 8) / (2 ** 16)), mode='RGB').convert("RGB")
    print("label", val_ds[idx]['label'])
    encoding = torch.stack(
        (val_transforms(img),)).to("cuda:0")
    print(encoding.shape)

    rgb_img = img.copy()
    tensor = encoding#.unsqueeze(0)

    # target_layers = [model.layer4[-1]]
    target_layers = [model.convnext.encoder.stages[-1].layers[-1]]
    input_tensor = tensor.cpu()#.cuda()
    cam = GradCAM(model=model.cpu(), target_layers=target_layers, use_cuda=False)
    # targets = [ClassifierOutputTarget(0)]
    targets = [ClassifierOutputTarget(1)]
    targets = None
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe
        self.transform = transform

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]#.to_numpy()
        image = row['image']
        image = image.point(lambda p: p*((2**8)/(2**16)), mode='RGB').convert("RGB")
        sample = self.transform(image)
        label = int(row['rural'])
        return sample, label

    def __len__(self):
        return len(self.dataframe)
    
    
    
def pytorch_train_model(dataloaders, dataset_sizes, device, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# define the LightningModule
class LitFineTune(pl.LightningModule):
    def __init__(self, model_ft):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_ft
        self.loss_module = torch.nn.CrossEntropyLoss()

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=4e-3)

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]



def lightning_model_train(model_ft, model_name, device, train_loader, val_loader, test_loader, save_name=None):
    CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/ConvNets")
    if save_name is None:
        save_name = model_name

    wandb.login()

    wandb_logger = WandbLogger(name=save_name, project=model_name)
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),  # Where to save models
        # We run on a single GPU (if possible)
        gpus=1 if str(device) == "cuda:0" else 0,
        # How many epochs to train for if no patience is set
        max_epochs=25,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, mode="max", monitor="val_acc"
            ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch"),
            # Log learning rate every epoch
            pl.callbacks.progress.TQDMProgressBar(refresh_rate=1),
        ],
        logger=[wandb_logger],
    )  # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need


    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = LitFineTune.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)  # To be reproducable
        model = LitFineTune(model_ft)
        trainer.fit(model, train_loader, val_loader)
        model = LitFineTune.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )  # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result


# # init the autoencoder
# autoencoder = LitAutoEncoder(encoder, decoder)


def pytorch_finetune(train_df, val_df, test_df):
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # data_dir = 'data/hymenoptera_data'
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
    #                                         data_transforms[x])
    #                 for x in ['train', 'val']}
    
    image_datasets = {"train": CustomDataset(train_df, data_transforms['train']),
                      "val": CustomDataset(val_df, data_transforms['val']),
                      "test": CustomDataset(test_df, data_transforms['val'])}

    batch_size = 64

    dataloaders = {"train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=batch_size,
                                                shuffle=True, num_workers=4),
                   "val": torch.utils.data.DataLoader(image_datasets["val"], batch_size=batch_size,
                                                shuffle=False, num_workers=4),
                   "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=batch_size,
                                                shuffle=False, num_workers=4),}

    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
    #                                             shuffle=True, num_workers=16)
    #             for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = {0: "urban", 1: "rural"} #image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # model_ft = models.resnet18(pretrained=True)
    model_ft = models.convnext_tiny(weights=models.convnext.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    finetuned = torch.load("./convnext-tiny-finetuned-mri/pytorch_model.bin")

    num_ftrs = model_ft.classifier[-1].in_features
    # num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    # model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft.classifier[-1] = torch.nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    save_name = f"test_trainer_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    lightning_model_train(model_ft, "ConvNext", device, dataloaders['train'], dataloaders['val'], dataloaders['test'], save_name=save_name)

    return

    criterion = torch.nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = pytorch_train_model(dataloaders, dataset_sizes, device, model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)


def process_grad_cam(model, test_df, preprocess_val_one):
    from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    # model.config_class.update({"use_return_dict": False})


    # target_layers = [model.layer4[-1]]
    target_layers = [model.convnext.encoder.stages[-1].layers[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    # targets = [ClassifierOutputTarget(0)]

    def wrapper(func):
        def model_forward_decorator(*args, **kargs):
            res = func(*args, **kargs)
            return res.logits
        return model_forward_decorator

    model.forward = wrapper(model.forward)


    # idx = 161
    # image = test_df.iloc[idx]['image']
    def get_visualization(row):
        image = row['image']
        rgb_img = image.point(lambda p: p * ((2 ** 8) / (2 ** 16)), mode='RGB').convert("RGB")
        rgb_img = rgb_img.resize((224, 224))
        rgb_img = np.float32(rgb_img) / 255
        label = int(row['rural'])
    # print("label", label)
        encoding = torch.stack((preprocess_val_one(image),)).to("cuda:0")

        tensor = encoding  # .unsqueeze(0)


        input_tensor = tensor.cuda()
        targets = [ClassifierOutputTarget(label)]
        # targets = [HuggingFaceClassifierOutputTarget(1)]
        # targets = None
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        return visualization

    test_df['grad_cam_visualizations'] = test_df.apply(get_visualization, axis=1)
    print(1)
    # imgplot = plt.imshow(visualization)
    # plt.show()


# class RandomGaussianBlur2d(torch.nn.Module):
#     ...
#     def __init__(self,
#                  kernel_size: typing.Tuple[int, int],
#                  sigma: typing.Tuple[float, float],
#                  border_type: str = 'reflect',
#                  separable: bool = True) -> None:
#         super().__init__()
#         self.kernel_size: typing.Tuple[int, int] = kernel_size
#         self.sigma: typing.Tuple[float, float] = sigma
#         ...
#
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         sigma = torch.empty(1).uniform_(sigma_min, sigma_max).item() # sample sigma for both axes
#         return gaussian_blur2d(input,
#                        self.kernel_size,
#                        (sigma, sigma),  # sigma sampled used
#                        self.border_type,
#                        self.separable)


class RandomMedianBlur(kornia.augmentation.IntensityAugmentationBase2D):
    """Add random blur with a box filter to an image tensor.

    .. image:: _static/img/RandomBoxBlur.png

    Args:
        kernel_size: the blurring kernel size.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``constant``, ``reflect``, ``replicate`` or ``circular``.
        normalized: if True, L1 norm of the kernel is set to 1.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    .. note::
        This function internally uses :func:`kornia.filters.box_blur`.

    Examples:
        >>> img = torch.ones(1, 1, 24, 24)
        >>> out = RandomMedianBlur((7, 7))(img)
        >>> out.shape
        torch.Size([1, 1, 24, 24])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomMedianBlur((7, 7), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        kernel_size: typing.Tuple[int, int] = (3, 3),
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
        return_transform: typing.Optional[bool] = None,
    ) -> None:
        super().__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim
        )
        self.flags = dict(kernel_size=kernel_size)

    def compute_transformation(self, input: torch.Tensor, params: typing.Dict[str, torch.Tensor], flags: typing.Dict[str, typing.Any]) -> torch.Tensor:
        return self.identity_matrix(input)

    def apply_transform(
        self, input: torch.Tensor, params: typing.Dict[str, torch.Tensor], flags: typing.Dict[str, typing.Any], transform: typing.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return kornia.filters.median.median_blur(input, flags["kernel_size"])


class Identity:  # used for skipping transforms
    def __call__(self, im):
        return im



class PreprocessStage:
    
    def __init__(self, labels):
        label2id, id2label = dict(), dict()
        for i, label in enumerate(labels):
            label2id[label] = i
            id2label[i] = label
        # Load model to fine-tune

        feature_extractor = AutoFeatureExtractor.from_pretrained("convnext-tiny-finetuned-mri")
        self.feature_extractor = feature_extractor

        self.model = AutoModelForImageClassification.from_pretrained("convnext-tiny-finetuned-mri",
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
            )

        
        # Define preprocess
        normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
        self.train_transforms = Compose(
                [
                    Resize(feature_extractor.size),
                    RandomResizedCrop(feature_extractor.size, scale=(0.08, 2.0), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomChoice([
                        transforms.RandomApply([transforms.RandomRotation((90, 90))]),
                        transforms.RandomApply([transforms.RandomRotation((-90, -90))]),
                    ]),
                    ToTensor(),
                    # kornia.augmentation.RandomMotionBlur(3, 35., 0.5, keepdim=True),
                    transforms.RandomChoice([
                        kornia.augmentation.RandomPerspective(keepdim=True, p=1.0),
                        kornia.augmentation.RandomThinPlateSpline(keepdim=True, p=1.0),
                        Identity()
                    ]),
                    transforms.RandomChoice([
                        # kornia.augmentation.RandomBoxBlur(kernel_size=(1, 1), keepdim=True, p=1.0),
                        kornia.augmentation.RandomBoxBlur(kernel_size=(3, 3), keepdim=True, p=1.0),
                        kornia.augmentation.RandomBoxBlur(kernel_size=(5, 5), keepdim=True, p=1.0),
                        kornia.augmentation.RandomBoxBlur(kernel_size=(7, 7), keepdim=True, p=1.0),
                        kornia.augmentation.RandomBoxBlur(kernel_size=(9, 9), keepdim=True, p=1.0),
                        # RandomMedianBlur(kernel_size=(1, 1), keepdim=True, p=1.0),
                        RandomMedianBlur(kernel_size=(3, 3), keepdim=True, p=1.0),
                        RandomMedianBlur(kernel_size=(5, 5), keepdim=True, p=1.0),
                        RandomMedianBlur(kernel_size=(7, 7), keepdim=True, p=1.0),
                        RandomMedianBlur(kernel_size=(9, 9), keepdim=True, p=1.0),
                        RandomMedianBlur(kernel_size=(11, 11), keepdim=True, p=1.0),
                        # kornia.augmentation.RandomGaussianBlur(kernel_size=(1, 1), sigma=(0.1, 2.0), keepdim=True, p=1.0),
                        kornia.augmentation.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), keepdim=True, p=1.0),
                        kornia.augmentation.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), keepdim=True, p=1.0),
                        kornia.augmentation.RandomGaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0), keepdim=True, p=1.0),
                        kornia.augmentation.RandomGaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.0), keepdim=True, p=1.0),
                        kornia.augmentation.RandomGaussianBlur(kernel_size=(11, 11), sigma=(0.1, 2.0), keepdim=True, p=1.0),
                        Identity()
                    ]),
                    transforms.RandomChoice([
                        kornia.augmentation.RandomMotionBlur(3, 35., 0.5, keepdim=True, p=1.0),
                        Identity()
                    ], p=[0.1, 0.9]),
                    # kornia.filters.median.MedianBlur((3, 3)),
                    normalize,
                ]
            )

        self.val_transforms = Compose(
                [
                    Resize(feature_extractor.size, interpolation=transforms.InterpolationMode.BICUBIC),
                    CenterCrop(feature_extractor.size),
                    ToTensor(),
                    normalize,
                ]
            )

    def preprocess_train(self, example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [
            self.train_transforms(image.point(lambda p: p*((2**8)/(2**16)), mode='RGB').convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch

    def preprocess_val_one(self, image):
        return self.val_transforms(image.point(lambda p: p*((2**8)/(2**16)), mode='RGB').convert("RGB"))

    def preprocess_val(self, example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [self.preprocess_val_one(image) for image in example_batch["image"]]
        return example_batch



def create_dataset(preprocess: PreprocessStage, train_df, val_df):
    
    
    # Cast to dataset
    train_dataset = Dataset.from_dict({'image': train_df['image'].values.tolist(), 'label': train_df['rural'].astype(int).values.tolist()})
    train_dataset = train_dataset.cast_column('label', ClassLabel(names=['Urban', 'Rural']))
    val_dataset = Dataset.from_dict({'image': val_df['image'].values.tolist(), 'label': val_df['rural'].astype(int).values.tolist()})
    val_dataset = val_dataset.cast_column('label', ClassLabel(names=['Urban', 'Rural']))
    
    # Train/val split
    # splits = dataset1.train_test_split(test_size=0.2, stratify_by_column="label")
    splits = DatasetDict({"train": train_dataset, "test": val_dataset})

    labels = splits["train"].features["label"].names


    train_ds = splits['train']
    val_ds = splits['test']
    train_ds.set_transform(preprocess.preprocess_train)
    val_ds.set_transform(preprocess.preprocess_val)

    train_ds = CustomDataset(train_df, preprocess.train_transforms)
    val_ds = CustomDataset(val_df, preprocess.val_transforms)

    return train_ds, val_ds


def load_best_model(path):
    dirs = glob.glob(path + "*" + os.path.sep)
    def read_trainer_state(d):
        f_path = os.path.join(d, "trainer_state.json")
        with open(f_path, "r") as f:
            state = json.load(f)
            return state
    states = list(map(read_trainer_state, dirs))
    best_state = min(states, key=lambda x: x['best_metric'])
    best_state_checkpoint = best_state['best_model_checkpoint']
    print(f"loaded best checkpoint {best_state_checkpoint} with loss {best_state['best_metric']}")
    model = AutoModelForImageClassification.from_pretrained(best_state_checkpoint).to("cuda:0")
    return model
    

def load_data():
    kfold_splits = 10
    
    # Scan data
    data_path = r"C:\Users\gkove\Downloads\Risk-taking Urban bats vs. hesitant rural bats_ urbanization drives behavioral flexibility and brain plasticity in Egyptian fruit bats\sorted_data"
    meta = scan_sorted_data(data_path)
    print("scanned data")
    

    # print(torch.__version__)
    df = load_images(meta, data_path)
    df = df.query("after == False and image_index >= 19 and image_index <= 81")
    # df = df.query("after == False")
    print("loaded images")

    # Train/test split
    rng = np.random.default_rng(seed=42)
    batsies = df[['rural', 'bat_id']].drop_duplicates().reset_index()
    smallest_group = batsies.groupby("rural").size().min()
    balanced_batsies = batsies.groupby("rural")[['bat_id', 'rural']].sample(n=smallest_group, random_state=rng).reset_index()
    
    # def train_val_test_split(lst):
    #     s = np.array(lst)
    #     rng.shuffle(s)
    #     kfold = KFold(5, shuffle=False)
    #     train_val, test = np.split(s, [int(len(s)*0.9)])
    #     list(kfold.split(train_val))[fold_index]
    #     return pd.Series({"train": train, "val": val, "test": test})
    
    preprocess_stage = PreprocessStage([False, True])
    print("loaded preprocess stage")
    
    random_state = np.random.RandomState(43)
    train_val_0, test_0 = train_test_split(balanced_batsies[['bat_id', 'rural']], train_size=0.9, stratify=balanced_batsies['rural'], shuffle=True, random_state=random_state)
    
    test_bats = test_0['bat_id'].values
    test_df = df[df['bat_id'].isin(test_bats)]
    
    
    skf = StratifiedKFold(n_splits=kfold_splits, shuffle=False)
    def kfold_generator():
        for fold_idx, (train_index, val_index) in enumerate(skf.split(train_val_0['bat_id'], train_val_0['rural'])):
            train_0 = train_val_0.iloc[train_index]
            val_0 = train_val_0.iloc[val_index]
            
            train_bats = train_0['bat_id'].values
            val_bats = val_0['bat_id'].values

            train_df = df[df['bat_id'].isin(train_bats)]
            val_df = df[df['bat_id'].isin(val_bats)]
            yield fold_idx, (train_df, val_df)
    
    return meta, df, balanced_batsies, preprocess_stage, train_val_0, test_df, kfold_generator()


def main():
    wakepy.set_keepawake(keep_screen_awake=False)
    
    meta, df, balanced_batsies, preprocess_stage, train_val_0, test_df, kfold_generator = load_data()
    
    train = False
    if train:
    
        
        wandb.login()        
        
        output_dir = f"test_trainer_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        for fold_idx, (train_df, val_df) in kfold_generator:
        
            # train_0, val_0 = train_test_split(train_val_0, train_size=0.7, stratify=train_val_0['rural'], random_state=random_state)

            
            
            # balanced_batsies_split = balanced_batsies.groupby(["rural"])['bat_id'].unique().apply(train_val_test_split)
            # train_bats = rng.choice(bat_ids, size=int(len(bat_ids)*0.9), replace=False)
            # train_bats = np.concatenate(balanced_batsies_split['train'].values)
            # val_bats = np.concatenate(balanced_batsies_split['val'].values)
            # test_bats = np.concatenate(balanced_batsies_split['test'].values)
            

            print(f"Done train/val/test split fold {fold_idx}")
            # pytorch_finetune(train_df, val_df, test_df)
            
            # return
            
            train_ds, val_ds = create_dataset(preprocess_stage, train_df, val_df)
            _ = run_train(train_ds, val_ds, preprocess_stage.model, preprocess_stage.feature_extractor, output_dir + f"_{fold_idx}")
            model = load_best_model(output_dir)
    else:
                
        # model = AutoModelForImageClassification.from_pretrained("test_trainer_2022-07-25_19-56-17").to("cuda:0"
        # model = AutoModelForImageClassification.from_pretrained("test_trainer_2022-08-02_22-46-41").to("cuda:0")
        # model = load_best_model("test_trainer_2022-09-03_18-47-35")
        model = load_best_model("test_trainer_2022-09-03_18-47-35_1")


    run_test(test_df, model, preprocess_stage.preprocess_val_one)
    # run_test(pd.concat([val_df, test_df]), model, preprocess_val_one)

    wakepy.unset_keepawake()
    
    

if __name__ == "__main__":
    main()
