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
import argparse

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
        if arr.max() == 0:
            image = np.array(arr).astype("uint16")
            img = Image.fromarray(image)
            return img
        # The data is assumed to be between 0 and 2**16, but some records are between 0 and 1, so scale those to 0 to 2**16
        if 0 < arr.max() <= 1:
            arr = arr * 2**15
        if arr.max() < 1:
            image = np.array(arr).astype("uint16")
            img = Image.fromarray(image)
            return img
        image = np.array(arr).astype("uint16")
        assert image.max() > 0, (image.max(), arr.max())
        img = Image.fromarray(image)
        img = img.rotate(90)
        return img
    def flatten_3d_image(row):
        mat = scipy.io.loadmat(os.path.join(data_path, row['full_path'].iloc[0]))
        volume = mat.get('Data', mat.get('img'))
        start_range = 0
        end_range = volume.shape[2]
        return pd.DataFrame({"image": [gm_to_img(volume[:, :, i]) for i in range(start_range, end_range)]})
    
    df = pd.merge(meta, meta.query("contrast == 'GM'").groupby('fname').apply(flatten_3d_image).reset_index().rename(columns={"level_1":"image_index"}), on="fname")
    
    return df


def collate_fn(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


class TrainerWithPersistentWorkers(Trainer):
    
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
    trainer = TrainerWithPersistentWorkers(
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

    
    def grp_pred(grp):
        return np.round(grp.mean())
    test_pred: pd.DataFrame() = df2.groupby(["bat_id", "repetition"])[['rural', 'pred']].agg({"rural": 'mean', "pred": grp_pred})
    test_accuracy = (test_pred["rural"] == test_pred['pred']).mean()
    print(test_accuracy, test_pred["rural"].sum())
    print(test_pred.to_markdown())



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
    

def load_data(data_path, kfold_splits):
    kfold_splits = 10
    
    # Scan data
    meta = scan_sorted_data(data_path)
    print("scanned data")
    

    df = load_images(meta, data_path)
    df = df.query("after == False and image_index >= 19 and image_index <= 81")
    print("loaded images")

    # Train/test split
    rng = np.random.default_rng(seed=42)
    batsies = df[['rural', 'bat_id']].drop_duplicates().reset_index()
    smallest_group = batsies.groupby("rural").size().min()
    balanced_batsies = batsies.groupby("rural")[['bat_id', 'rural']].sample(n=smallest_group, random_state=rng).reset_index()
    
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


def main(data_path, training_output_dir, kfold_splits, evaluate_model):
    wakepy.set_keepawake(keep_screen_awake=False)
    
    meta, df, balanced_batsies, preprocess_stage, train_val_0, test_df, kfold_generator = load_data(data_path, kfold_splits)
    
    train = evaluate_model is None
    if train:
    
        
        wandb.login()        
        
        output_dir = training_output_dir + f"test_trainer_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        for fold_idx, (train_df, val_df) in kfold_generator:

            print(f"Done train/val/test split fold {fold_idx}")

            
            train_ds, val_ds = create_dataset(preprocess_stage, train_df, val_df)
            _ = run_train(train_ds, val_ds, preprocess_stage.model, preprocess_stage.feature_extractor, output_dir + f"_{fold_idx}")
            model = load_best_model(output_dir)
    else:
        model = load_best_model(evaluate_model)

    run_test(test_df, model, preprocess_stage.preprocess_val_one)

    wakepy.unset_keepawake()
    
    

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description='Rural vs urban bats model training')
    parser.add_argument('-data','--data-path', help='Path to sorted_data directory in dataset', required=True)
    parser.add_argument('-output','--output-dir', help='Path to output location', required=True)
    parser.add_argument('-k','--k-fold', help='K for K-fold', type=int, default=10)
    parser.add_argument('-evaluate','--evaluate-model', help='Path to (cross-validation) models for evaluation only', required=False)

    args = parser.parse_args()
    
    data_path = getattr(args, 'data_path')
    training_output_dir = getattr(args, 'output_dir')
    kfold_splits = getattr(args, 'k_fold')
    evaluate_model = getattr(args, 'evaluate_model')
    
    main(data_path, training_output_dir, kfold_splits, evaluate_model)
