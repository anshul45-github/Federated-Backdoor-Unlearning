"""Select healing clients using class recall or a matched random control."""

from collections import Counter
from copy import copy
from random import Random

import torch
from torch.utils.data import DataLoader, Subset

from src.training.train_utils import evaluate_extended


class ClassBalancedHealing:
    def __init__(self, partitions, clients, evaluation_transform, batch_size, clients_per_round,
                 validation_per_class, seed, random_client_seed=None):
        self.clients_per_round = clients_per_round
        self.history = []
        self.client_rng = None if random_client_seed is None else Random(random_client_seed)
        self.criterion = torch.nn.CrossEntropyLoss()
        dataset = partitions[clients[0]].dataset
        labels = dataset.targets
        class_indices = [[] for _ in range(10)]
        for client in clients:
            for index in partitions[client].indices:
                class_indices[labels[index]].append(index)
        random = Random(seed)
        self.validation_indices = []
        for label, indices in enumerate(class_indices):
            if len(indices) <= validation_per_class:
                raise ValueError(f"Class {label} needs more than {validation_per_class} retained examples")
            self.validation_indices.extend(random.sample(sorted(indices), validation_per_class))
        held_out = set(self.validation_indices)
        validation_dataset = copy(dataset)
        validation_dataset.transform = evaluation_transform
        self.validation = DataLoader(Subset(validation_dataset, self.validation_indices), batch_size=batch_size)
        self.partitions = list(partitions)
        self.class_counts = {}
        for client in clients:
            indices = [index for index in partitions[client].indices if index not in held_out]
            self.partitions[client] = Subset(dataset, indices)
            if indices:
                counts = Counter(labels[index] for index in indices)
                self.class_counts[client] = [counts[label] for label in range(10)]
        if len(self.class_counts) < clients_per_round:
            raise ValueError("Too few retained clients with healing data for the requested round size")

    def measure(self, model, device):
        metrics = evaluate_extended(model, self.validation, self.criterion, device)
        return {
            "clean_accuracy_pct": metrics["accuracy"],
            "per_class_recall": [metrics["per_class"][f"class_{label}"]["recall"] for label in range(10)],
        }

    def __call__(self, round_idx, model, device):
        validation = self.measure(model, device)
        deficits = [1 - recall for recall in validation["per_class_recall"]]
        scores = {client: sum(count * deficit for count, deficit in zip(counts, deficits))
                  for client, counts in self.class_counts.items()}
        if self.client_rng is None:
            selected = sorted(scores, key=lambda client: (-scores[client], client))[:self.clients_per_round]
        else:
            selected = self.client_rng.sample(sorted(scores), self.clients_per_round)
        total = sum(len(self.partitions[client]) for client in selected)
        self.history.append({
            "round": round_idx, "validation_before_round": validation, "client_scores": scores,
            "selected_clients": selected,
            "fedavg_weights": {client: len(self.partitions[client]) / total for client in selected},
        })
        print(f"Healing round {round_idx}: selected clients {selected}", flush=True)
        return selected
