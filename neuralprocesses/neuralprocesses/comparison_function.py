import lab as B
import torch
from typing import Tuple, Union, Optional

__all__ = ["distance", "difference", "rotate"]


def distance(relational_encoding_type,
             xc,
             yc,
             xt,
             net,
             relational_out_dim: int,
             sparse: bool = False,
             k: Optional[int] = None,
             non_equivariant_dim: Optional[list] = None,
             ):
    batch_size, set_size, feature_dim = xc.shape
    _, target_set_size, _ = xt.shape
    _, _, out_dim = yc.shape

    if non_equivariant_dim is not None:
        all_dim = set(range(feature_dim))
        xc_non_equivariant = xc[:, :, non_equivariant_dim]
        xt_non_equivariant = xt[:, :, non_equivariant_dim]

        equivariant_dim = list(set(all_dim) - set(non_equivariant_dim))
        xc = xc[:, :, equivariant_dim]
        xt = xt[:, :, equivariant_dim]
        len_non_equivariant_dim = len(non_equivariant_dim)
    else:
        len_non_equivariant_dim = 0

    if relational_encoding_type == "simple":
        if sparse and set_size > k:
            distance_matrix = B.sqrt(B.sum((xt.unsqueeze(2) - xc.unsqueeze(1)) ** 2, axis=-1))
            _, nearest_indices = distance_matrix.topk(k, dim=2, largest=False)
            batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2).expand(-1, target_set_size, k)
            target_indices = torch.arange(target_set_size).unsqueeze(1).expand(-1, k).unsqueeze(
                0).repeat(batch_size, 1, 1)

            if non_equivariant_dim is not None:
                dist_x = B.concat(
                    distance_matrix.unsqueeze(-1),
                    xc_non_equivariant.unsqueeze(1).repeat(1, target_set_size, 1, 1),
                    yc.unsqueeze(1).repeat(1, target_set_size, 1, 1),
                    axis=-1,
                )
            else:
                dist_x = B.concat(distance_matrix.unsqueeze(-1),
                                yc.unsqueeze(1).repeat(1, target_set_size, 1, 1), axis=-1)

            dist_x = dist_x[batch_indices, target_indices, nearest_indices]
        else:
            # (batch_size, target_set_size, set_size, 1))
            dist_x = B.sqrt(B.sum((xt.unsqueeze(2) - xc.unsqueeze(1)) ** 2, axis=-1).unsqueeze(-1))

            if non_equivariant_dim is not None:
                dist_x = B.concat(
                    dist_x,
                    xc_non_equivariant.unsqueeze(1).repeat(1, target_set_size, 1, 1),
                    yc.unsqueeze(1).repeat(1, target_set_size, 1, 1),
                    axis=-1,
                )
            else:
                # (batch_size, target_set_size, set_size, 2))
                dist_x = B.concat(dist_x, yc.unsqueeze(1).repeat(1, target_set_size, 1, 1), axis=-1)

        # (batch_size, target_set_size * set_size, 2))
        dist_x = dist_x.reshape(batch_size, -1, 1 + out_dim + len_non_equivariant_dim)
        batch_size, diff_size, filter_size = dist_x.shape

        # (batch_size * target_set_size * set_size, 2))
        x = dist_x.view(batch_size * diff_size, filter_size)

    elif relational_encoding_type == "full":
        if sparse and set_size > k:
            relational_matrix = B.sqrt(
                B.sum((xc.unsqueeze(2) - xc.unsqueeze(1)) ** 2, axis=-1).unsqueeze(-1))
            yc_matrix_1 = yc.unsqueeze(2).repeat(1, 1, set_size, 1)
            yc_matrix_2 = yc.unsqueeze(1).repeat(1, set_size, 1, 1)
            # shape: [batch_size, target_set_size, set_size, set_size, 3]
            relational_matrix = B.concat(
                relational_matrix, yc_matrix_1, yc_matrix_2, axis=-1
            ).unsqueeze(1).repeat(1, target_set_size, 1, 1, 1)

            distance_matrix = B.sqrt(
                B.sum((xt.unsqueeze(2) - xc.unsqueeze(1)) ** 2, axis=-1))

            _, nearest_indices = distance_matrix.topk(k, dim=2, largest=False)

            batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, target_set_size,
                                                                                                   k, k)
            target_indices = torch.arange(target_set_size).unsqueeze(1).unsqueeze(2).expand(-1, k,
                                                                                            k).unsqueeze(0).repeat(
                batch_size, 1, 1, 1)
            row_indices = nearest_indices.unsqueeze(2).expand(-1, -1, k, k)
            col_indices = nearest_indices.unsqueeze(3).expand(-1, -1, k, k)

            dist_x = distance_matrix.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, set_size, 1)
            if non_equivariant_dim is not None:
                dist_x = B.concat(
                    dist_x,
                    relational_matrix,
                    xc_non_equivariant.unsqueeze(1).repeat(
                        1, target_set_size, set_size, 1
                    ),
                    axis=-1,
                )
            else:
                dist_x = B.concat(
                    dist_x,
                    relational_matrix,
                    axis=-1,
                )

            dist_x = dist_x[batch_indices, target_indices, row_indices, col_indices]

            dist_x = dist_x.reshape(
                batch_size, target_set_size * k * k, -1
            )

        else:
            relational_matrix = B.sqrt(B.sum((xc.unsqueeze(2) - xc.unsqueeze(1)) ** 2, axis=-1).unsqueeze(-1))

            yc_matrix_1 = yc.unsqueeze(2).repeat(1, 1, set_size, 1)
            yc_matrix_2 = yc.unsqueeze(1).repeat(1, set_size, 1, 1)
            # shape: [batch_size, set_size, set_size, 3]
            relational_matrix = B.concat(
                relational_matrix, yc_matrix_1, yc_matrix_2, axis=-1
            )
            # shape: [batch_size, set_size * set_size, 3]
            relational_matrix = relational_matrix.reshape(
                batch_size, set_size * set_size, -1
            )

            # shape: [batch_size, target_set_size, set_size * set_size, 1]
            dist_x = B.sqrt(
                B.sum((xt.unsqueeze(2) - xc.unsqueeze(1)) ** 2, axis=-1).unsqueeze(
                    -1
                )
            ).repeat(1, 1, set_size, 1)
            if non_equivariant_dim is not None:
                dist_x = B.concat(
                    dist_x,
                    relational_matrix.unsqueeze(1).repeat(1, target_set_size, 1, 1),
                    xc_non_equivariant.unsqueeze(1).repeat(
                        1, target_set_size, set_size, 1
                    ),
                    axis=-1,
                )
            else:
                # shape: [batch_size, target_set_size, set_size * set_size, 4]
                dist_x = B.concat(
                    dist_x,
                    relational_matrix.unsqueeze(1).repeat(1, target_set_size, 1, 1),
                    axis=-1,
                )

            dist_x = dist_x.reshape(
                batch_size, target_set_size * set_size * set_size, -1
            )

        batch_size, diff_size, filter_size = dist_x.shape
        # x shape: [batch_size * target_set_size * set_size * set_size, 4]
        x = dist_x.view(batch_size * diff_size, filter_size)
    else:
        raise NotImplementedError

    x = net(x)
    x = x.view(batch_size, diff_size, relational_out_dim)
    encoded_feature_dim = x.shape[-1]

    if relational_encoding_type == "simple":
        if sparse and set_size > k:
            set_size_new = k
        else:
            set_size_new = set_size
    elif relational_encoding_type == "full":
        if sparse and set_size > k:
            set_size_new = k * k
        else:
            set_size_new = set_size * set_size
    else:
        raise NotImplementedError

    x = x.view(batch_size, target_set_size, set_size_new, encoded_feature_dim)
    encoded_target_x = x.sum(dim=2)
    if non_equivariant_dim is not None:
        encoded_target_x = B.concat(encoded_target_x, xt_non_equivariant, axis=-1)

    encoded_target_x = B.transpose(encoded_target_x)
    return encoded_target_x


def difference(relational_encoding_type,
             xc,
             yc,
             xt,
             net,
             relational_out_dim: int,
             sparse: bool = False,
             k: Optional[int] = None,
             non_equivariant_dim: Optional[list] = None,
             ):
    batch_size, set_size, feature_dim = xc.shape
    _, target_set_size, _ = xt.shape
    _, _, out_dim = yc.shape

    if non_equivariant_dim is not None:
        all_dim = set(range(feature_dim))
        xc_non_equivariant = xc[:, :, non_equivariant_dim]
        xt_non_equivariant = xt[:, :, non_equivariant_dim]

        equivariant_dim = list(set(all_dim) - set(non_equivariant_dim))
        xc = xc[:, :, equivariant_dim]
        xt = xt[:, :, equivariant_dim]
        len_non_equivariant_dim = len(non_equivariant_dim)
    else:
        len_non_equivariant_dim = 0

    if relational_encoding_type == "simple":
        if sparse and set_size > k:
            distance_matrix = B.sqrt(
                B.sum((xt.unsqueeze(2) - xc.unsqueeze(1)) ** 2, axis=-1))
            _, nearest_indices = distance_matrix.topk(k, dim=2, largest=False)

            batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2).expand(-1, target_set_size, k)
            target_indices = torch.arange(target_set_size).unsqueeze(1).expand(-1, k).unsqueeze(
                0).repeat(batch_size, 1, 1)

            xc_pairs = B.concat(xc, yc, axis=-1).unsqueeze(1)

            xt_pairs = B.concat(
                xt,
                B.cast(xt.dtype, B.zeros(batch_size, target_set_size, out_dim)),
                axis=-1,
            ).unsqueeze(2)

            if non_equivariant_dim is not None:
                diff_x = B.concat(
                    xt_pairs - xc_pairs,
                    xc_non_equivariant.unsqueeze(1).repeat(1, target_set_size, 1, 1),
                    axis=-1)
            else:
                diff_x = xt_pairs - xc_pairs
            diff_x = diff_x[batch_indices, target_indices, nearest_indices].reshape(
                        batch_size, -1, feature_dim + out_dim
                    )

        else:
            xc_pairs = B.concat(xc, yc, axis=-1).unsqueeze(1)
            xt_pairs = B.concat(
                xt,
                B.cast(xt.dtype, B.zeros(batch_size, target_set_size, out_dim)),
                axis=-1,
            ).unsqueeze(2)

            if non_equivariant_dim is not None:
                diff_x = B.concat(
                    xt_pairs - xc_pairs,
                    xc_non_equivariant.unsqueeze(1).repeat(1, target_set_size, 1, 1),
                    axis=-1)
            else:
                diff_x = xt_pairs - xc_pairs
            diff_x = diff_x.reshape(batch_size, -1, feature_dim + out_dim)

        batch_size, diff_size, filter_size = diff_x.shape
        x = diff_x.view(batch_size * diff_size, filter_size)

    elif relational_encoding_type == "full":
        if sparse and set_size > k:
            raise NotImplementedError("Sparse FullRCNP is not implemented yet")

        else:
            relational_matrix = xc.unsqueeze(2) - xc.unsqueeze(1)
            yc_matrix_1 = yc.unsqueeze(2).repeat(1, 1, set_size, 1)
            yc_matrix_2 = yc.unsqueeze(1).repeat(1, set_size, 1, 1)
            # shape: [batch_size, set_size, set_size, dim_x+2*dim_y]
            relational_matrix = B.concat(
                relational_matrix, yc_matrix_1, yc_matrix_2, axis=-1
            )
            # shape: [batch_size, set_size * set_size, dim_x+2*dim_y]
            relational_matrix = relational_matrix.reshape(
                batch_size, set_size * set_size, -1
            )

            context_xp = xc.unsqueeze(1)
            target_xp = xt.unsqueeze(2)
            # shape: [batch_size, target_set_size, set_size * set_size, dim_x]
            diff_x = (target_xp - context_xp).repeat(1, 1, set_size, 1)

            if non_equivariant_dim is not None:
                diff_x = B.concat(
                    diff_x,
                    relational_matrix.unsqueeze(1).repeat(1, target_set_size, 1, 1),
                    xc_non_equivariant.unsqueeze(1).repeat(
                        1, target_set_size, set_size, 1
                    ),
                    axis=-1,
                )
            else:
                # shape: [batch_size, target_set_size, set_size * set_size, 2*dim_x+2*dim_y]
                diff_x = B.concat(
                    diff_x,
                    relational_matrix.unsqueeze(1).repeat(1, target_set_size, 1, 1),
                    axis=-1,
                )

            diff_x = diff_x.reshape(
                batch_size, target_set_size * set_size * set_size, -1
            )
            batch_size, diff_size, filter_size = diff_x.shape
            # x shape: [batch_size * target_set_size * set_size * set_size, 4]
            x = diff_x.view(batch_size * diff_size, filter_size)
    else:
        raise NotImplementedError

    x = net(x)
    x = x.view(batch_size, diff_size, relational_out_dim)
    encoded_feature_dim = x.shape[-1]

    if relational_encoding_type == "simple":
        if sparse and set_size > k:
            set_size_new = k
        else:
            set_size_new = set_size
    elif relational_encoding_type == "full":
        if sparse and set_size > k:
            set_size_new = k * k
        else:
            set_size_new = set_size * set_size
    else:
        raise NotImplementedError

    x = x.view(batch_size, target_set_size, set_size_new, encoded_feature_dim)
    encoded_target_x = x.sum(dim=2)
    if non_equivariant_dim is not None:
        encoded_target_x = B.concat(encoded_target_x, xt_non_equivariant, axis=-1)

    encoded_target_x = B.transpose(encoded_target_x)
    return encoded_target_x


def rotate(relational_encoding_type,
         xc,
         yc,
         xt,
         net,
         relational_out_dim: int,
         sparse: bool = False,
         k: Optional[int] = None,
         non_equivariant_dim: Optional[list] = None,
         ):
    raise NotImplementedError