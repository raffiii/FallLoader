from datasets import SuperSet
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


if __name__ == "__main__":
    video = list(np.random.randint(0, 255, (16, 3, 720, 1080)))
    test_dataset_with_mae(video)
