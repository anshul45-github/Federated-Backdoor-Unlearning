import torch
from torch.utils.data import Dataset, Subset
from typing import List, Optional, Tuple

class PoisonedDataset(Dataset):
    """
    A dataset wrapper that applies a backdoor trigger to a deterministic subset of samples.
    """
    def __init__(self, base_dataset: Dataset, poison_indices: List[int], target_class: int, trigger_size: int = 3,
                 trigger_type: str = "white_patch"):
        super().__init__()
        if trigger_type not in {"white_patch", "checkerboard"}:
            raise ValueError(f"Unknown trigger_type: {trigger_type}")
        if not isinstance(trigger_size, int) or trigger_size <= 0:
            raise ValueError("trigger_size must be a positive integer")
        self.base_dataset = base_dataset
        self.poison_indices = set(poison_indices)
        self.target_class = target_class
        self.trigger_size = trigger_size
        self.trigger_type = trigger_type
        
        # CIFAR-10 Normalization constants used in our data pipeline
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        self.std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        
        # Calculate the normalized value for "white" (1.0)
        self.trigger_value = (1.0 - self.mean) / self.std
        if trigger_type == "checkerboard":
            # Top-left cell is white; alternate black/white in both directions.
            positions = torch.arange(trigger_size)
            pixels = ((positions[:, None] + positions[None, :]) % 2 == 0).float()
            self.trigger_patch = (pixels.unsqueeze(0) - self.mean) / self.std
        else:
            # Keep the original broadcasted white-patch behavior by default.
            self.trigger_patch = self.trigger_value

    def __len__(self):
        return len(self.base_dataset)

    def apply_trigger(self, img: torch.Tensor) -> torch.Tensor:
        """Apply the selected normalized trigger after augmentation, at bottom right."""
        img_poisoned = img.clone()
        c, h, w = img_poisoned.shape
        if self.trigger_size > min(h, w):
            raise ValueError("trigger_size exceeds the image dimensions")
        patch = self.trigger_patch.to(device=img.device, dtype=img.dtype)
        img_poisoned[:, h - self.trigger_size:, w - self.trigger_size:] = patch
        return img_poisoned

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.base_dataset[idx]
        
        if idx in self.poison_indices:
            img = self.apply_trigger(img)
            label = self.target_class
            
        return img, label

def create_asr_dataloader(clean_test_loader: torch.utils.data.DataLoader, target_class: int, trigger_size: int = 3,
                          batch_size: int = 64, trigger_type: str = "white_patch") -> torch.utils.data.DataLoader:
    """
    Creates an Attack Success Rate (ASR) dataloader from the clean test loader.
    1. Filters out original target-class samples.
    2. Applies the trigger to 100% of the remaining samples.
    3. Changes their labels to the target class.
    """
    base_dataset = clean_test_loader.dataset
    
    # 1. Filter out samples that are originally the target class
    valid_indices = []
    # If it's a torchvision dataset, labels are usually accessible via .targets
    # If it's a Subset or FakeData, we might need to iterate.
    # We will try .targets first for speed.
    has_targets = hasattr(base_dataset, 'targets')
    
    if has_targets:
        targets = base_dataset.targets
        for i in range(len(base_dataset)):
            label = targets[i]
            if isinstance(label, torch.Tensor):
                label = label.item()
            if label != target_class:
                valid_indices.append(i)
    else:
        # Fallback (slower, but works for Subset/FakeData)
        for i in range(len(base_dataset)):
            _, label = base_dataset[i]
            if label != target_class:
                valid_indices.append(i)
                
    # Create a subset of valid (non-target-class) samples
    filtered_dataset = Subset(base_dataset, valid_indices)
    
    # 2. Poison 100% of these valid samples
    # PoisonedDataset takes indices relative to its base_dataset (which is now filtered_dataset)
    # So all indices from 0 to len(filtered_dataset)-1 should be poisoned
    all_indices = list(range(len(filtered_dataset)))
    asr_dataset = PoisonedDataset(filtered_dataset, all_indices, target_class, trigger_size, trigger_type)
    
    # Return loader
    return torch.utils.data.DataLoader(asr_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
