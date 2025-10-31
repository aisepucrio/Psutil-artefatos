from codecarbon import EmissionsTracker
import psutil
import csv
import os
import medmnist
import logging
import torch
from datasets import Dataset, DatasetDict
from medmnist import INFO
from collections import Counter
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip
import torchvision.transforms as transforms
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from transformers import AutoTokenizer
from flwr_datasets.partitioner import PathologicalPartitioner
from flwr_datasets.partitioner import ShardPartitioner
from functools import partial


# Start CodeCarbon tracker with project name and collect initial system metrics
tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\TraceFL\tracefl\dataset_preparation.py")
tracker.start()

mem_start = psutil.virtual_memory().used / (1024**2)
cpu_start = psutil.cpu_percent(interval=None)


class DatasetLoader:
    """Handles dataset loading and conversion."""

    @staticmethod
    def _medmnist_to_hf_dataset(medmnist_dataset):
        data_dict = {"image": [], "label": []}
        for pixels, label in medmnist_dataset:
            data_dict["image"].append(pixels)
            data_dict["label"].append(label.item())
        return Dataset.from_dict(data_dict)

    @classmethod
    def load_medmnist(cls, data_flag='pathmnist', download=True):
        """Loads and converts a MedMNIST dataset to a Hugging Face DatasetDict."""
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        train_dataset = DataClass(split='train', download=download)
        test_dataset = DataClass(split='test', download=download)
        hf_train_dataset = cls._medmnist_to_hf_dataset(train_dataset)
        hf_test_dataset = cls._medmnist_to_hf_dataset(test_dataset)
        hf_dataset = DatasetDict({"train": hf_train_dataset, "test": hf_test_dataset})
        logging.info(f'conversion to hf_dataset done')
        return hf_dataset


class TokenizerFactory:
    """Creates tokenizer functions based on the dataset type."""

    @staticmethod
    def _create_default_tokenizer(tokenizer, input_col_name):
        def tokenize_function(examples):
            return tokenizer(examples[input_col_name], truncation=True, padding='max_length', max_length=128)
        return tokenize_function

    @staticmethod
    def _create_yahoo_answers_tokenizer(tokenizer):
        def tokenize_function(examples):
            examples['label'] = examples['topic']
            return tokenizer(examples['question_title'] + ' ' + examples['question_content'], truncation=True, padding='max_length', max_length=128)
        return tokenize_function

    @classmethod
    def create_tokenizer(cls, cfg):
        """Creates and returns a tokenizer function."""
        tokenizer = AutoTokenizer.from_pretrained(cfg.mname, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if cfg.dname == "yahoo_answers_topics":
            return cls._create_yahoo_answers_tokenizer(tokenizer)
        input_col_name = "content" if cfg.dname == "dbpedia_14" else "text"
        return cls._create_default_tokenizer(tokenizer, input_col_name)


class TransformsFactory:
    """Creates train and test transformation functions for image datasets."""

    @staticmethod
    def _create_cifar10_transforms():
        def apply_train_transform(example):
            transform = Compose([
                Resize((32, 32)),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            example['pixel_values'] = [transform(image.convert("RGB")) for image in example['img']]
            example['label'] = torch.tensor(example['label'])
            del example['img']
            return example

        def apply_test_transform(example):
            transform = Compose([
                Resize((32, 32)),
                ToTensor(),
                Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            example['pixel_values'] = [transform(image.convert("RGB")) for image in example['img']]
            example['label'] = torch.tensor(example['label'])
            del example['img']
            return example
        return {'train': apply_train_transform, 'test': apply_test_transform}

    @staticmethod
    def _create_mnist_transforms():
        def apply_train_transform(example):
            transform = Compose([
                Resize((32, 32)),
                ToTensor(),
                Normalize((0.1307,), (0.3081,))
            ])
            example['pixel_values'] = [transform(image.convert("RGB")) for image in example['image']]
            example['label'] = torch.tensor(example['label'])
            return example

        def apply_test_transform(example):
            transform = Compose([
                Resize((32, 32)),
                ToTensor(),
                Normalize((0.1307,), (0.3081,))
            ])
            example['pixel_values'] = [transform(image.convert("RGB")) for image in example['image']]
            example['label'] = torch.tensor(example['label'])
            del example['image']
            return example
        return {'train': apply_train_transform, 'test': apply_test_transform}

    @staticmethod
    def _create_pathmnist_organmnist_transforms():
        tfms = transforms.Compose([Resize((32, 32)), ToTensor(), Normalize(mean=[.5], std=[.5])])

        def apply_transform(example):
            example['pixel_values'] = [tfms(image.convert('RGB')) for image in example['image']]
            example['label'] = torch.tensor(example['label'])
            del example['image']
            return example
        return {'train': apply_transform, 'test': apply_transform}

    @classmethod
    def create_transforms(cls, cfg):
        """Creates and returns train and test transformation functions."""
        if cfg.dname == "cifar10":
            return cls._create_cifar10_transforms()
        elif cfg.dname == "mnist":
            return cls._create_mnist_transforms()
        elif cfg.dname in ['pathmnist', 'organamnist']:
            return cls._create_pathmnist_organmnist_transforms()
        else:
            raise ValueError(f"Unknown dataset: {cfg.dname}")


class DatasetProcessor:
    """Processes datasets by applying transformations and tokenization."""

    def __init__(self, cfg):
        self.cfg = cfg

    def _apply_image_transforms(self, dataset, transforms, batch_size=256, num_proc=8):
        """Applies image transformations to a dataset."""
        return dataset.map(transforms, batched=True, batch_size=batch_size, num_proc=num_proc).with_format("torch")

    def _apply_text_tokenization(self, dataset, tokenize_function):
        """Applies tokenization to a dataset."""
        return dataset.map(tokenize_function)

    def process_image_dataset(self, dataset, transforms):
        """Processes an image dataset with transformations."""
        return {
            "train": self._apply_image_transforms(dataset['train'], transforms['train']),
            "test": self._apply_image_transforms(dataset['test'], transforms['test'])
        }

    def process_transformer_dataset(self, dataset, tokenize_function):
        """Processes a transformer dataset with tokenization."""
        return {
            "train": self._apply_text_tokenization(dataset['train'], tokenize_function),
            "test": self._apply_text_tokenization(dataset['test'], tokenize_function)
        }


def get_labels_count(partition, target_label_col):
    """Counts the occurrences of each label in a dataset partition."""
    label2count = Counter(example[target_label_col] for example in partition)
    return dict(label2count)


def fix_partition(cfg, c_partition, target_label_col):
    """Cleans and truncates a client data partition."""
    label2count = get_labels_count(c_partition, target_label_col)
    filtered_labels = {label: count for label, count in label2count.items() if count >= 10}
    indices_to_select = [i for i, example in enumerate(c_partition) if example[target_label_col] in filtered_labels]
    ds = c_partition.select(indices_to_select)
    assert cfg.max_per_client_data_size > 0, f"max_per_client_data_size: {cfg.max_per_client_data_size}"
    if len(ds) > cfg.max_per_client_data_size:
        ds = ds.select(range(cfg.max_per_client_data_size))
    if len(ds) % cfg.batch_size == 1:
        ds = ds.select(range(len(ds) - 1))
    partition_labels_count = get_labels_count(ds, target_label_col)
    return {'partition': ds, 'partition_labels_count': partition_labels_count}


class DataPartitioner:
    """Abstract base class for data partitioning strategies."""

    def __init__(self, cfg, target_label_col, fetch_only_test_data):
        self.cfg = cfg
        self.target_label_col = target_label_col
        self.fetch_only_test_data = fetch_only_test_data
        self.fds = None
        self.server_data = None
        self.client2data = {}
        self.client2class = {}

    def _load_and_partition(self, subtask=None):
        """Loads and partitions the dataset."""
        raise NotImplementedError

    def partition(self, subtask=None):
        """Executes the partitioning process."""
        self._load_and_partition(subtask)
        return {
            'client2data': self.client2data,
            'server_data': self.server_data,
            'client2class': self.client2class,
            'fds': self.fds
        }


class DirichletDataPartitioner(DataPartitioner):
    """Partitions data using a Dirichlet distribution."""

    def __init__(self, cfg, target_label_col, fetch_only_test_data):
        super().__init__(cfg, target_label_col, fetch_only_test_data)
        self.partitioner = DirichletPartitioner(
            num_partitions=cfg.num_clients,
            partition_by=self.target_label_col,
            alpha=cfg.dirichlet_alpha,
            min_partition_size=0,
            self_balancing=True,
            shuffle=True,
        )

    def _load_and_partition(self, subtask=None):
        self._partition_helper(subtask)

    def _partition_helper(self, subtask):
        """Helper function to perform the partitioning."""
        clients_data = []
        clients_class = []
        if self.cfg.dname in ['pathmnist', 'organamnist']:
            hf_dataset = DatasetLoader.load_medmnist(data_flag=self.cfg.dname, download=True)
            self.partitioner.dataset = hf_dataset['train']
            self.fds = self.partitioner
            if self.cfg.max_server_data_size < len(hf_dataset['test']):
                self.server_data = hf_dataset['test'].select(range(self.cfg.max_server_data_size))
            else:
                self.server_data = hf_dataset['test']
        else:
            if subtask is not None:
                self.fds = FederatedDataset(dataset=self.cfg.dname, partitioners={"train": self.partitioner}, subset=subtask)
            else:
                self.fds = FederatedDataset(dataset=self.cfg.dname, partitioners={"train": self.partitioner})
            if len(self.fds.load_split("test")) < self.cfg.max_server_data_size:
                self.server_data = self.fds.load_split("test")
            else:
                self.server_data = self.fds.load_split("test").select(range(self.cfg.max_server_data_size))
        for cid in range(self.cfg.num_clients):
            client_partition = self.fds.load_partition(cid)
            temp_dict = {}
            if self.cfg.max_per_client_data_size > 0:
                temp_dict = fix_partition(self.cfg, client_partition, self.target_label_col)
            else:
                temp_dict = {'partition': client_partition, 'partition_labels_count': get_labels_count(client_partition, self.target_label_col)}
            if len(temp_dict['partition']) >= self.cfg.batch_size:
                clients_data.append(temp_dict['partition'])
                clients_class.append(temp_dict['partition_labels_count'])
        self.client2data = {f"{id}": v for id, v in enumerate(clients_data)}
        self.client2class = {f"{id}": v for id, v in enumerate(clients_class)}


class ShardedDataPartitioner(DataPartitioner):
    """Partitions data using a sharded non-IID distribution."""

    def __init__(self, cfg, target_label_col, fetch_only_test_data, num_classes_per_partition):
        super().__init__(cfg, target_label_col, fetch_only_test_data)
        self.num_classes_per_partition = num_classes_per_partition
        self.partitioner = ShardPartitioner(
            num_partitions=cfg.num_clients,
            partition_by=self.target_label_col,
            shard_size=2000,
            num_shards_per_partition=self.num_classes_per_partition,
            shuffle=True
        )

    def _load_and_partition(self, subtask=None):
        self._partition_helper(subtask)

    def _partition_helper(self, subtask):
        clients_data = []
        clients_class = []
        if self.cfg.dname in ['pathmnist', 'organamnist']:
            hf_dataset = DatasetLoader.load_medmnist(data_flag=self.cfg.dname, download=True)
            self.partitioner.dataset = hf_dataset['train']
            self.fds = self.partitioner
            if self.cfg.max_server_data_size < len(hf_dataset['test']):
                self.server_data = hf_dataset['test'].select(range(self.cfg.max_server_data_size))
            else:
                self.server_data = hf_dataset['test']
        else:
            if subtask is not None:
                self.fds = FederatedDataset(dataset=self.cfg.dname, partitioners={"train": self.partitioner}, subset=subtask)
            else:
                self.fds = FederatedDataset(dataset=self.cfg.dname, partitioners={"train": self.partitioner})
            if len(self.fds.load_split("test")) < self.cfg.max_server_data_size:
                self.server_data = self.fds.load_split("test")
            else:
                self.server_data = self.fds.load_split("test").select(range(self.cfg.max_server_data_size))
        for cid in range(self.cfg.num_clients):
            client_partition = self.fds.load_partition(cid)
            temp_dict = {}
            if self.cfg.max_per_client_data_size > 0:
                temp_dict = fix_partition(self.cfg, client_partition, self.target_label_col)
            else:
                temp_dict = {'partition': client_partition, 'partition_labels_count': get_labels_count(client_partition, self.target_label_col)}
            if len(temp_dict['partition']) >= self.cfg.batch_size:
                clients_data.append(temp_dict['partition'])
                clients_class.append(temp_dict['partition_labels_count'])
        self.client2data = {f"{id}": v for id, v in enumerate(clients_data)}
        self.client2class = {f"{id}": v for id, v in enumerate(clients_class)}


class PathologicalDataPartitioner(DataPartitioner):
    """Partitions data using a pathological (highly non-IID) strategy."""

    def __init__(self, cfg, target_label_col, fetch_only_test_data, num_classes_per_partition):
        super().__init__(cfg, target_label_col, fetch_only_test_data)
        self.num_classes_per_partition = num_classes_per_partition
        self.partitioner = PathologicalPartitioner(
            num_partitions=cfg.num_clients,
            partition_by=self.target_label_col,
            num_classes_per_partition=self.num_classes_per_partition,
            shuffle=True,
            class_assignment_mode='deterministic'
        )

    def _load_and_partition(self, subtask=None):
        self._partition_helper(subtask)

    def _partition_helper(self, subtask):
        clients_data = []
        clients_class = []
        if self.cfg.dname in ['pathmnist', 'organamnist']:
            hf_dataset = DatasetLoader.load_medmnist(data_flag=self.cfg.dname, download=True)
            self.partitioner.dataset = hf_dataset['train']
            self.fds = self.partitioner
            if self.cfg.max_server_data_size < len(hf_dataset['test']):
                self.server_data = hf_dataset['test'].select(range(self.cfg.max_server_data_size))
            else:
                self.server_data = hf_dataset['test']
        else:
            if subtask is not None:
                self.fds = FederatedDataset(dataset=self.cfg.dname, partitioners={"train": self.partitioner}, subset=subtask)
            else:
                self.fds = FederatedDataset(dataset=self.cfg.dname, partitioners={"train": self.partitioner})
            if len(self.fds.load_split("test")) < self.cfg.max_server_data_size:
                self.server_data = self.fds.load_split("test")
            else:
                self.server_data = self.fds.load_split("test").select(range(self.cfg.max_server_data_size))
        for cid in range(self.cfg.num_clients):
            client_partition = self.fds.load_partition(cid)
            temp_dict = {}
            if self.cfg.max_per_client_data_size > 0:
                temp_dict = fix_partition(self.cfg, client_partition, self.target_label_col)
            else:
                temp_dict = {'partition': client_partition, 'partition_labels_count': get_labels_count(client_partition, self.target_label_col)}
            if len(temp_dict['partition']) >= self.cfg.batch_size:
                clients_data.append(temp_dict['partition'])
                clients_class.append(temp_dict['partition_labels_count'])
        self.client2data = {f"{id}": v for id, v in enumerate(clients_data)}
        self.client2class = {f"{id}": v for id, v in enumerate(clients_class)}


class ClientsAndServerDatasets:
    """Prepares and manages datasets for clients and the server."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.client2data = {}
        self.server_testdata = None
        self.client2class = {}
        self.fds = None
        self._setup()

    def _get_partitioner(self):
        """Returns the appropriate partitioner based on the configuration."""
        dist_type = self.cfg.data_dist.dist_type
        target_label_col = "label"
        if self.cfg.dname == "yahoo_answers_topics":
            target_label_col = "topic"
        fetch_only_test_data = False  # Changed this to a constant value
        if dist_type == 'non_iid_dirichlet':
            return DirichletDataPartitioner(self.cfg, target_label_col, fetch_only_test_data)
        elif dist_type.startswith('sharded-non-iid'):
            num_classes = int(dist_type.split('-')[-1])
            return ShardedDataPartitioner(self.cfg, target_label_col, fetch_only_test_data, num_classes)
        elif dist_type.startswith('PathologicalPartitioner'):
            num_classes = int(dist_type.split('-')[-1])
            return PathologicalDataPartitioner(self.cfg, target_label_col, fetch_only_test_data, num_classes)
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

    def _setup_hugging_dataset(self):
        """Sets up the Hugging Face dataset."""
        partitioner = self._get_partitioner()
        partitioned_data = partitioner.partition()
        self.client2data = partitioned_data["client2data"]
        self.server_testdata = partitioned_data["server_data"]
        self.client2class = partitioned_data["client2class"]
        self.fds = partitioned_data["fds"]
        logging.info(f"client2class: {self.client2class}")
        logging.info(f"> client2class {self.client2class}")
        data_per_client = [len(dl) for dl in self.client2data.values()]
        logging.info(f"Data per client in experiment {data_per_client}")
        min_data = min(len(dl) for dl in self.client2data.values())
        logging.info(f"Min data on a client: {min_data}")

        if self.cfg.dname in ["cifar10", "mnist", 'pathmnist', 'organamnist']:
            transforms = TransformsFactory.create_transforms(cfg=self.cfg)
            dataset_processor = DatasetProcessor(self.cfg)
            processed_data = {
                "client2data": {k: dataset_processor.process_image_dataset(
                    {"train": v, "test": self.server_testdata}, transforms)["train"]
                    for k, v in self.client2data.items()
                },
                "server_testdata": dataset_processor.process_image_dataset(
                    {"train": self.server_testdata, "test": self.server_testdata}, transforms
                )["test"]
            }
            self.client2data = processed_data["client2data"]
            self.server_testdata = processed_data["server_testdata"]
        elif self.cfg.dname in ['dbpedia_14', 'yahoo_answers_topics']:
            tokenize_function = TokenizerFactory.create_tokenizer(self.cfg)
            dataset_processor = DatasetProcessor(self.cfg)
            processed_data = {
                "client2data": {k: dataset_processor.process_transformer_dataset(
                    {"train": v, "test": self.server_testdata}, tokenize_function)["train"]
                    for k, v in self.client2data.items()
                },
                "server_testdata": dataset_processor.process_transformer_dataset(
                    {"train": self.server_testdata, "test": self.server_testdata}, tokenize_function
                )["test"]
            }
            self.client2data = processed_data["client2data"]
            self.server_testdata = processed_data["server_testdata"]

    def _setup(self):
        """Sets up the datasets."""
        self._setup_hugging_dataset()

    def get_data(self):
        """Returns the prepared client and server datasets."""
        return {
            "server_testdata": self.server_testdata,
            "client2class": self.client2class,
            "client2data": self.client2data,
            "fds": self.fds
        }


# Collect final system metrics and stop tracker
mem_end = psutil.virtual_memory().used / (1024**2)
cpu_end = psutil.cpu_percent(interval=None)

# Save psutil data to psutil_data.csv
csv_file = "psutil_data.csv"
file_exists = os.path.isfile(csv_file)
with open(csv_file, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(["file", "mem_start_MB", "mem_end_MB", "mem_diff_MB", "cpu_start_percent", "cpu_end_percent"])
    writer.writerow([
        __file__,
        f"{mem_start:.2f}",
        f"{mem_end:.2f}",
        f"{mem_end - mem_start:.2f}",
        f"{cpu_start:.2f}",
        f"{cpu_end:.2f}"
    ])

tracker.stop()
