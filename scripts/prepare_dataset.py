from datasets import load_dataset
from pathlib import Path

def prepare():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    processed_dir = data_dir / "processed"

    dataset = load_dataset("imagefolder", data_dir=str(data_dir), cache_dir=str(processed_dir))
    print(dataset)

prepare()
