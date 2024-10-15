import torch
import torch.nn.functional as F
import pdb

from config import cfg

def DAGPromptEvaluator(loader, gnn, prompt, center_embeddings, device):
    prompt.eval()
    correct = 0
    for batch in loader:
        batch = batch.to(device)
        node_index_saves = batch.node_index_saves if cfg.model.adaptive_adj else None

        out = gnn(batch.x, batch.edge_index, batch.batch, prompt, 'dagprompt', 
                  node_index_saves=node_index_saves)
        # )
        # pdb.set_trace()
        accumulated_similarity_matrix = torch.zeros(out.size(1), center_embeddings.size(1)).to(device)
        for i, (embedding, center_embedding) in enumerate(zip(out, center_embeddings)):
            similarity_matrix = F.cosine_similarity(embedding.unsqueeze(1), center_embedding.unsqueeze(0), dim=-1)
            # Use the gamma to control the weight of each hop
            accumulated_similarity_matrix += similarity_matrix * prompt.gamma[i]

        pred = accumulated_similarity_matrix.argmax(dim=1)
        correct += int((pred == batch.y).sum())

    # print([x.item() for x in gnn.global_edge_weights.values()])
    acc = correct / len(loader.dataset)
    return acc
