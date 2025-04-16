"""Helper functions for NeatNet."""
import torch
import torch.nn as nn
import gensim


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_word_embedding_model():
    """This is a FastText model pre-trained on Common Crawl: see paper for details.
    Run this command to download the vector file:
        wget "https://drive.google.com/uc?export=download&id=1vkxBzqhQzqr39P4go7QXbyTi2OAkjlo2" -O "crawl-300d-tidying-subword.vec"
    """
    we_model = gensim.models.KeyedVectors.load_word2vec_format("crawl-300d-tidying-subword.vec", binary=False, unicode_errors='replace')

    # Generating vectors for composite words which will be useful later.
    we_model['coffee_table'] = we_model['coffee'] + we_model['table']

    # Here you can downsample from the word embedding vectors, if you would like to speed up training.
    # There is also a FastText script to help reduce the dimension: https://fasttext.cc/docs/en/crawl-vectors.html
    num_items = len(we_model.vocab)
    reduce_factor = 5
    we_model_dim = 300
    normed_model_dim = int(we_model_dim / reduce_factor)
    word_dim = normed_model_dim
    normed_model = torch.zeros((num_items, normed_model_dim))
    items = list(we_model.vocab.keys())
    for i, item in enumerate(items):
        vector = torch.from_numpy(we_model[item])
        reduced_vector = vector[::reduce_factor]
        vector_norm = reduced_vector.norm(p=2, dim=0, keepdim=True)
        normed_vector = reduced_vector.div(vector_norm)
        normed_model[i] = normed_vector

    print('Loaded FastText model')
    return normed_model, word_dim


def loss_vae(pred_pos, true_pos, mu, log_var, vae_beta):
    """Computes the total loss for the VAE model."""
    pred_loss_func = nn.MSELoss(reduction="mean")
    loss_pred = pred_loss_func(pred_pos, true_pos)

    # Mean across dimensions.
    loss_kl = 0.5 * torch.mean(mu.pow(2) + log_var.exp() - 1 - log_var)
    loss_total = loss_pred + vae_beta * loss_kl
    return loss_total, loss_pred, loss_kl


# Note: use max instead of getting last value if ordering of scene_ids not guaranteed (from 0 upwards).
# Makes scene ids unique across all scenes in batch.
# Used by global pooling function to generate graph encoding vector for each scene in batch.
def create_unique_scene_ids(batch_ids, scene_ids):
    """Creates unique scene ids for each scene in the batch."""
    num_users = batch_ids[-1].item() + 1
    unique_scene_ids = torch.empty(batch_ids.size(), dtype=torch.long).to(DEVICE)

    scenes_so_far = 0
    nodes_so_far = 0
    for user_id in torch.arange(num_users):
        user_mask = batch_ids == user_id
        user_scene_ids = torch.masked_select(scene_ids, user_mask)
        unique_scene_ids[nodes_so_far : nodes_so_far + user_scene_ids.size(0)] = (
            user_scene_ids + scenes_so_far
        )
        scenes_so_far += user_scene_ids[-1].item() + 1
        nodes_so_far += user_scene_ids.size(0)
    return unique_scene_ids
