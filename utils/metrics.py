import torch


def recall_score(y_true: torch.Tensor, y_pred: torch.Tensor, top_k: int = 10):
    """
    Args:
        y_true (Tensor): shape (batch_size, items_total)
        y_pred (Tensor): shape (batch_size, items_total)
        top_k (int):
    Returns:
        output (batch_size, )
    """
    # predict_indices, shape (batch_size, top_k)
    _, predict_indices = y_pred.topk(k=top_k)
    predict, truth = y_pred.new_zeros(y_pred.shape).scatter_(dim=1, index=predict_indices,
                                                             value=1).long(), y_true.long()
    tp, t = ((predict == truth) & (truth == 1)).sum(dim=-1), truth.sum(dim=-1)
    return tp.float() / t.float()


def dcg(y_true: torch.Tensor, y_pred: torch.Tensor, top_k: int = 10):
    """
    Args:
        y_true: (batch_size, items_total)
        y_pred: (batch_size, items_total)
        top_k (int):

    Returns:
            (batch_size, )
    """
    # predict_indices, shape (batch_size, top_k)
    _, predict_indices = y_pred.topk(k=top_k)
    gain = y_true.gather(-1, predict_indices)  # (batch_size, top_k)
    return (gain.float() / torch.log2(torch.arange(top_k, device=y_pred.device).float() + 2)).sum(dim=-1)  # (batch_size, )


def ndcg_score(y_true: torch.Tensor, y_pred: torch.Tensor, top_k: int = 10):
    """
    Args:
        y_true: (batch_size, items_total)
        y_pred: (batch_size, items_total)
        top_k (int):
    Returns:
            (batch_size, )
    """
    dcg_score = dcg(y_true, y_pred, top_k)
    idcg_score = dcg(y_true, y_true, top_k)
    return dcg_score / idcg_score


def PHR(y_true: torch.Tensor, y_pred: torch.Tensor, top_k: int = 10):
    """
    Args:
        y_true (Tensor): shape (batch_size, items_total)
        y_pred (Tensor): shape (batch_size, items_total)
        top_k (int):
    Returns:
        output (batch_size, )
    """
    # predict_indices, shape (batch_size, top_k)
    _, predict_indices = y_pred.topk(k=top_k)
    predict, truth = y_pred.new_zeros(y_pred.shape).scatter_(dim=1, index=predict_indices,
                                                             value=1).long(), y_true.long()
    return torch.mul(predict, truth).sum(dim=1)


def get_metric(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
        Args:
            y_true: tensor (samples_num, items_total)
            y_pred: tensor (samples_num, items_total)
        Returns:
            scores: dict, key -> metric name, value -> metric Tensor
    """

    result = {}
    for top_k in [10, 20, 30, 40]:
        result.update({
            f'recall_{top_k}': recall_score(y_true, y_pred, top_k=top_k),
            f'ndcg_{top_k}': ndcg_score(y_true, y_pred, top_k=top_k),
            f'PHR_{top_k}': PHR(y_true, y_pred, top_k=top_k)
        })
    return result


def get_all_metric(metric_list):
    """
    get all metrics based on metric_list
    :param metric_list: [{"recall_10": Tensor, "ndcg_10": Tensor, "PHR_10": Tensor, ...},...]
    :return:
        res: {"recall_10": float, "ndcg_10": float, "PHR_10": float ...}
    """
    metric_names = metric_list[0].keys()
    result = {}

    for metric_name in metric_names:
        concat_metric_tensor = torch.cat([metric[metric_name] for metric in metric_list], dim=0)
        if metric_name.startswith(f"PHR_"):
            result[metric_name] = concat_metric_tensor.nonzero().shape[0] / concat_metric_tensor.shape[0]
        else:
            result[metric_name] = concat_metric_tensor.mean().item()

    result = {key: float(f"{result[key]:.4f}") for key in result}
    result = sorted(result.items(), key=lambda item: item[0], reverse=False)
    result = {item[0]: float(f"{item[1]:.4f}") for item in result}

    return result
