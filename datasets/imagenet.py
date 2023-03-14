import os
import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets
import numpy as np

@DATASET_REGISTRY.register()
class ImageNet(DatasetBase):

    dataset_dir = "imagenet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            train = self.read_data(classnames, "train")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test = self.read_data(classnames, "val")

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)

        super().__init__(train_x=train, val=test, test=test)

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames, split_dir):
        # folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []
        label_mapper = {value: index for index, value in enumerate(classnames.values())}
        class_mapper = {index: value for index, value in enumerate(classnames.values())}
        if split_dir == 'train':
            split_dir = os.path.join(self.image_dir, split_dir)
            for imgpath in os.listdir(split_dir):
                class_name, _ = imgpath.split('_')
                classname = classnames[class_name]
                impath = os.path.join(split_dir, imgpath)
                label = label_mapper[classname]
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)
        if split_dir == 'val':
            split_dir = os.path.join(self.image_dir, split_dir)
            groundtruth = os.path.join(self.image_dir, 'ILSVRC2012_validation_ground_truth.txt')
            val_labels = []
            with open(groundtruth, 'r') as f:
                for l in f.readlines():
                    val_labels.append(int(l.strip()))
            val_labels = list(np.array(val_labels) - 1)
            for label, imgpath in zip(val_labels, os.listdir(split_dir)):
                impath = os.path.join(split_dir, imgpath)
                classname = class_mapper[label]
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)
        return items
