"""
GSM (Grade School Math) Dataset for Flow Matching

This module provides data loading utilities for the GSM dataset
in the flow matching task. Format: question||reasoning #### answer
"""

import re
from typing import Any, Dict, List

from torch.utils.data import DataLoader, Dataset


class GSMDataset(Dataset):
    """
    Dataset for GSM (Grade School Math) data.

    Each sample contains:
    - question: The math problem/question
    - reasoning_sequence: List of reasoning steps (e.g., ["2/2=1", "2+1=3"])
    - answer: The final answer (as a string)
    """

    def __init__(
        self,
        data_file: str,
        max_reasoning_length: int = 1000,
    ):
        """
        Initialize GSM dataset.

        Args:
            data_file: Path to GSM text file (one example per line)
            max_reasoning_length: Maximum number of reasoning tokens
        """
        self.data_file = data_file
        self.max_reasoning_length = max_reasoning_length

        # Load and parse data
        self.data = self._load_and_parse_data()

        print(f"Loaded {len(self.data)} examples from {data_file}")

    def _load_and_parse_data(self) -> List[Dict[str, Any]]:
        """Load and parse GSM data from text file."""
        data = []

        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Parse format: question||reasoning #### answer
                if '||' not in line or '####' not in line:
                    continue

                # Split by ||
                parts = line.split('||', 1)
                if len(parts) != 2:
                    continue

                question = parts[0].strip()
                rest = parts[1].strip()

                # Split by ####
                if '####' not in rest:
                    continue

                reasoning_part, answer = rest.rsplit('####', 1)
                reasoning_part = reasoning_part.strip()
                answer = answer.strip()

                # Extract reasoning steps (format: <<step1>> <<step2>> ...)
                reasoning_steps = re.findall(r'<<([^>]+)>>', reasoning_part)

                # Convert to list of strings
                reasoning_sequence = [step.strip() for step in reasoning_steps if step.strip()]

                # Skip if no reasoning steps
                if not reasoning_sequence:
                    continue

                # Estimate token count (simple approximation - will be tokenized properly by model)
                total_tokens = sum(len(step.split()) for step in reasoning_sequence)
                if total_tokens > self.max_reasoning_length:
                    continue

                data.append({
                    'question': question,
                    'reasoning_sequence': reasoning_sequence,
                    'answer': answer,
                    'num_reasoning_steps': len(reasoning_sequence),
                    'estimated_tokens': total_tokens
                })

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        sample = self.data[idx]

        return {
            'target_smiles': sample['question'],  # Map question to target_smiles
            'starting_materials': [sample['answer']],  # Map answer to starting_materials (as list)
            'reasoning_sequence': sample['reasoning_sequence'],  # Map reasoning steps
            'reaction_smiles': sample['reasoning_sequence'],  # Also provide as reaction_smiles for consistency
            'num_reasoning_steps': sample['num_reasoning_steps'],
            'estimated_tokens': sample['estimated_tokens']
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        num_steps = [item['num_reasoning_steps'] for item in self.data]
        token_counts = [item['estimated_tokens'] for item in self.data]

        return {
            'num_examples': len(self.data),
            'reasoning_steps_stats': {
                'mean': sum(num_steps) / len(num_steps) if num_steps else 0,
                'min': min(num_steps) if num_steps else 0,
                'max': max(num_steps) if num_steps else 0,
                'median': sorted(num_steps)[len(num_steps)//2] if num_steps else 0
            },
            'token_stats': {
                'mean': sum(token_counts) / len(token_counts) if token_counts else 0,
                'min': min(token_counts) if token_counts else 0,
                'max': max(token_counts) if token_counts else 0,
                'median': sorted(token_counts)[len(token_counts)//2] if token_counts else 0
            }
        }


def collate_gsm_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for GSM batch."""
    result = {
        'target_smiles': [item['target_smiles'] for item in batch],
        'starting_materials': [item['starting_materials'] for item in batch],
        'reasoning_sequence': [item['reasoning_sequence'] for item in batch],
        'reaction_smiles': [item['reaction_smiles'] for item in batch],
        'num_reasoning_steps': [item['num_reasoning_steps'] for item in batch],
        'estimated_tokens': [item['estimated_tokens'] for item in batch]
    }
    return result


def create_gsm_dataloader(
    data_file: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    max_reasoning_length: int = 1000,
    sampler=None
) -> DataLoader:
    """Create dataloader for GSM data."""

    dataset = GSMDataset(
        data_file=data_file,
        max_reasoning_length=max_reasoning_length,
    )

    # Use DistributedSampler if provided, otherwise use shuffle
    if sampler is not None:
        shuffle = False  # DistributedSampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_gsm_batch,
        pin_memory=True,  # Accelerate data transfer to GPU
        prefetch_factor=2,  # Prefetch batches for faster training
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive
    )


def calculate_gsm_reasoning_multiplier(
    data_file: str,
    tokenizer_name: str = "t5-base",
    coverage_percentile: float = 0.90
) -> int:
    """
    Calculate reasoning multiplier for GSM dataset.

    Args:
        data_file: Path to GSM data file
        tokenizer_name: Tokenizer to use for counting tokens
        coverage_percentile: Percentile for coverage (e.g., 0.90 = 90% coverage)

    Returns:
        Reasoning multiplier value
    """
    from transformers import AutoTokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load dataset
    dataset = GSMDataset(data_file=data_file, max_reasoning_length=10000)

    # Calculate token lengths for all reasoning sequences
    token_lengths = []
    for sample in dataset:
        reasoning_sequence = sample['reasoning_sequence']
        # Combine reasoning steps into a single string
        combined_text = " ".join(reasoning_sequence)
        # Tokenize
        tokens = tokenizer(combined_text, return_tensors="pt", add_special_tokens=False)
        token_lengths.append(tokens['input_ids'].shape[1])

    if not token_lengths:
        return 400  # Default fallback

    # Sort and find percentile
    sorted_lengths = sorted(token_lengths)
    percentile_idx = int(len(sorted_lengths) * coverage_percentile)
    max_length = sorted_lengths[percentile_idx] if percentile_idx < len(sorted_lengths) else sorted_lengths[-1]

    print("GSM Reasoning Statistics:")
    print(f"  Total examples: {len(token_lengths)}")
    print(f"  Mean tokens: {sum(token_lengths) / len(token_lengths):.1f}")
    print(f"  Min tokens: {min(token_lengths)}")
    print(f"  Max tokens: {max(token_lengths)}")
    print(f"  {coverage_percentile*100}% percentile: {max_length}")
    print(f"  Recommended multiplier: {max_length}")

    return max_length


def create_train_val_split_gsm(
    data_file: str,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> tuple[DataLoader, DataLoader]:
    """Create train/validation split for GSM data."""

    # Load full dataset
    dataset = GSMDataset(data_file=data_file, max_reasoning_length=kwargs.get('max_reasoning_length', 1000))

    # Split indices
    total_size = len(dataset)
    split_idx = int(total_size * train_ratio)

    # Create train and val datasets
    train_indices = list(range(split_idx))
    val_indices = list(range(split_idx, total_size))

    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_gsm_batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_gsm_batch
    )

    return train_loader, val_loader


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Test dataset
    dataset = GSMDataset(
        data_file="<CHECKPOINT_ROOT>/data/gsm_aug/test.txt",
        max_reasoning_length=1000,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Statistics: {dataset.get_statistics()}")

    # Test dataloader
    dataloader = create_gsm_dataloader(
        data_file="<CHECKPOINT_ROOT>/data/gsm_aug/test.txt",
        batch_size=4,
        shuffle=True
    )

    # Test batch
    for batch in dataloader:
        print(f"Batch keys: {batch.keys()}")
        print(f"Question: {batch['target_smiles'][0]}")
        print(f"Answer: {batch['starting_materials'][0]}")
        print(f"Reasoning sequence: {batch['reasoning_sequence'][0]}")
        break

    # Calculate reasoning multiplier
    multiplier = calculate_gsm_reasoning_multiplier(
        data_file="<CHECKPOINT_ROOT>/data/gsm_aug/test.txt",
        tokenizer_name="t5-base"
    )
    print(f"\nRecommended reasoning multiplier: {multiplier}")
