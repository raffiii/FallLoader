import falldata
from videoutils import sample_frames
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import torch


def test_dataset_with_mae(video):

    processor = VideoMAEImageProcessor.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics"
    )
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics"
    )

    inputs = processor(video, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])


def get_loader(path, filter=[]):
    dataset = falldata.SuperSet(
        path, "relative_superset.csv", processing=[sample_frames(16, 4)]
    )
    for f in filter:
        dataset = dataset.filter(f)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        collate_fn=falldata.collate_tuple,
    )
    return loader


if __name__ == "__main__":
    video = np.random.randint(0, 255, (16, 3, 480, 720))
    test_dataset_with_mae(list(video))
    loader = get_loader(
        "/home/rflbr/projects/Studium/MA/test_data",
        filter=[falldata.path_exist_filter, falldata.take_first_n(10)],
    )
    for batch in loader:
        videos, labels = batch
        print(len(videos), len(labels))
        print(videos[0].shape, labels[0])
        video = videos[0].transpose((0, 3, 1, 2))
        print(video.shape)
        test_dataset_with_mae(list(video))
