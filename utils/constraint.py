import torch


def constraint(device, prompt):
    """
    Calculate the constraint value for a given device and prompt.

    Parameters:
    - device (torch.device): The device to perform the calculations on.
    - prompt (torch.Tensor or list of torch.Tensor): The prompt(s) to calculate the constraint value for.

    Returns:
    - float: The constraint value.

    If the prompt is a list of tensors, the constraint value is calculated as the average of the constraint values for each prompt tensor.
    Otherwise, the constraint value is calculated for the single prompt tensor.

    The constraint value is calculated as the Frobenius norm of the difference between the prompt tensor(s) and the identity matrix,
    after performing matrix multiplication and transposition.

    Note: The prompt tensor(s) should have shape (n, n), where n is the number of dimensions.
    """
    if isinstance(prompt, list):
        total = 0
        for p in prompt:
            total = total + torch.norm(torch.mm(p, p.T) - torch.eye(p.shape[0]).to(device))
        return total / len(prompt)
    else:
        return torch.norm(torch.mm(prompt, prompt.T) - torch.eye(prompt.shape[0]).to(device))

