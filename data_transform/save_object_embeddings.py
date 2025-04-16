import csv
from pathlib import Path
from absl import app, flags
import torch
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

flags.DEFINE_string(
    "model", "all-MiniLM-L6-v2",
    "Name of the SentenceTransformer language model for text embeddings."
)
flags.DEFINE_string(
    "destination", None, "Path to save the object embeddings."
)
FLAGS = flags.FLAGS

# Constants.
OBJ_LABEL_FILE = "labels/object_labels_2024_02_29.csv"


class SentenceTransformerEmbeddingGenerator():
    def __init__(self, model_name):
        super().__init__()
        self.sentence_model = SentenceTransformer(model_name).to("cuda")

    def __call__(self, text):
        return self.sentence_model.encode(
            [text], batch_size=1, convert_to_tensor=True, device="cuda"
        ).detach()


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Load object labels.
    parent_path = Path(__file__).resolve().parents[1]
    with open(parent_path / OBJ_LABEL_FILE, "r") as fcsv:
        csv_data = csv.DictReader(fcsv)
        object_list = list(row["text"] for row in csv_data)

    # Generate embeddings.
    embedding_generator = SentenceTransformerEmbeddingGenerator(FLAGS.model)
    embeddings = torch.stack([embedding_generator(text).ravel() for text in object_list])
    print(f"Embedding shape: {embeddings[0].size()}")
    destination = Path(FLAGS.destination)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(zip(object_list, embeddings)), destination / f"object_embeddings_{FLAGS.model}.pt"
    )

    # Visualize dot product of embeddings.
    dotprod = torch.matmul(embeddings, embeddings.T)
    plt.figure(figsize=(32, 32))
    plt.imshow(dotprod.cpu().numpy())
    plt.xticks(range(len(object_list)), object_list, rotation=90)
    plt.yticks(range(len(object_list)), object_list)
    plt.savefig(f"../debug/{FLAGS.model}.png")


if __name__ == "__main__":
    app.run(main)
