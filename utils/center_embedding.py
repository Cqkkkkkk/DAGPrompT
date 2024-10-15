import pdb
import torch



def center_embedding(input, index, label_num):
    """
    Compute the center embeddings for each class.

    Args:
        input (torch.Tensor): The input embeddings.
        index (torch.Tensor): The class indices for each embedding.
        label_num (int): The total number of classes.

    Returns:
        torch.Tensor: The center embeddings for each class.
        torch.Tensor: The count of samples in each class.
    """
    device = input.device
    centers = torch.zeros(label_num, input.size(1)).to(device)
    centers = centers.scatter_add_(
        dim=0, 
        index=index.unsqueeze(1).expand(-1, input.size(1)), 
        src=input
    )
    # pdb.set_trace()
    # class_counts = torch.bincount(index, minlength=label_num).unsqueeze(1).to(dtype=input.dtype, device=device)

    # Determinstic version of bincount
    index_onehot = torch.nn.functional.one_hot(index, num_classes=label_num).to(dtype=input.dtype, device=device)
    class_counts = index_onehot.sum(dim=0).unsqueeze(1)

    # Take the average embeddings for each class
    # If directly divided the variable 'c', maybe encountering zero values in 'class_counts', such as the class_counts=[[0.],[4.]]
    # So we need to judge every value in 'class_counts' one by one, and separately divided them.
    # output_c = c/class_counts
    
    for i in range(label_num):
        if (class_counts[i].item() == 0):
            continue
        else:
            centers[i] /= (class_counts[i])
    return centers, class_counts


def center_embedding_multihop(input, index, label_num, hop_num):
    """
    Compute the center embeddings (for each-hop) for each class.

    Args:
        input (torch.Tensor): The input embeddings.
        index (torch.Tensor): The class indices for each embedding.
        label_num (int): The total number of classes.

    Returns:
        torch.Tensor: The center embeddings for each class.
        torch.Tensor: The count of samples in each class.
    """
    assert hop_num == input.size(0)
    device = input.device
    centers = torch.zeros(hop_num, label_num, input.size(-1)).to(device)
    for i in range(hop_num):
        centers[i] = centers[i].scatter_add_(
            dim=0, 
            index=index.unsqueeze(1).expand(-1, input.size(-1)), 
            src=input[i]
        )

    # class_counts = torch.bincount(index, minlength=label_num).unsqueeze(1).to(dtype=input.dtype, device=device)

    # Determinstic version of bincount
    index_onehot = torch.nn.functional.one_hot(index, num_classes=label_num).to(dtype=input.dtype, device=device)
    class_counts = index_onehot.sum(dim=0).unsqueeze(1)

    for i in range(hop_num):
        for j in range(label_num):
            if (class_counts[j].item() == 0):
                continue
            else:
                centers[i][j] /= (class_counts[j])
    return centers, class_counts


def distance2center(x, center):
    """
    Calculates the distance between each input vector and the center vectors.

    Args:
        x (torch.Tensor): Input tensor of shape (n, d), where n is the number of input vectors and d is the dimensionality of each vector.
        center (torch.Tensor): Center tensor of shape (k, d), where k is the number of center vectors and d is the dimensionality of each vector.

    Returns:
        torch.Tensor: Distance tensor of shape (n, k), where each element represents the distance between an input vector and a center vector.
    """
    n = x.size(0)
    k = center.size(0)
    input_power = torch.sum(x * x, dim=1, keepdim=True).expand(n, k)
    center_power = torch.sum(center * center, dim=1).expand(n, k)

    distance = input_power + center_power - 2 * \
        torch.mm(x, center.transpose(0, 1))
    return distance
