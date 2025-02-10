from torchvision import transforms as T
from config_utils import Args


def preprocess(examples, transform):
    examples["pixels"] = [transform(img) for img in examples["image"]]
    examples["labels"] = examples["label"]
    return examples


def get_transform(args: Args):
    if args.model_name == "lenet5":
        return T.Compose([
            T.Resize((32, 32)),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def test_unit():
    from datasets import load_dataset
    from functools import partial
    import torch

    transform = get_transform(args=Args(model_name="lenet5"))
    dataset = load_dataset("imagefolder", data_dir="data")
    dataset = dataset.map(partial(preprocess, transform=transform), batched=True, batch_size=None,
                          remove_columns=["image", "label"])

    print(dataset["train"])
    print(dataset["train"].features)
    data = dataset["train"][0]
    print(torch.Tensor(data["pixels"]).shape)
    print(data["labels"])
