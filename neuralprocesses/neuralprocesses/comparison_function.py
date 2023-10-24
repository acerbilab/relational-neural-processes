import lab as B
import torch
from typing import Tuple, Union, Optional

__all__ = ["distance", "difference", "rotate"]


def distance(relational_encoding_type,
             xc,
             yc,
             xt,
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
        batch_size, encoding_size, filter_size = dist_x.shape
        # (batch_size * target_set_size * set_size, 2))
        encoding = dist_x.view(batch_size * encoding_size, filter_size)

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

        batch_size, encoding_size, filter_size = dist_x.shape
        # x shape: [batch_size * target_set_size * set_size * set_size, 4]
        encoding = dist_x.view(batch_size * encoding_size, filter_size)
    else:
        raise NotImplementedError

    return encoding, encoding_size


def difference(relational_encoding_type,
             xc,
             yc,
             xt,
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
            diff_x = diff_x[batch_indices, target_indices, nearest_indices]

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
        batch_size, encoding_size, filter_size = diff_x.shape
        encoding = diff_x.view(batch_size * encoding_size, filter_size)

    elif relational_encoding_type == "full":
        if sparse and set_size > k:
            raise NotImplementedError("Sparse FullRCNP for difference comparison function is not implemented yet")

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
        batch_size, encoding_size, filter_size = diff_x.shape
        # x shape: [batch_size * target_set_size * set_size * set_size, 4]
        encoding = diff_x.view(batch_size * encoding_size, filter_size)
    else:
        raise NotImplementedError

    return encoding, encoding_size


def rotate(relational_encoding_type,
         xc,
         yc,
         xt,
         sparse: bool = False,
         k: Optional[int] = None,
         non_equivariant_dim: Optional[list] = None,
         ):

    batch_size, set_size, feature_dim = xc.shape
    _, target_set_size, _ = xt.shape
    _, _, out_dim = yc.shape

    if non_equivariant_dim is not None:
        # TODO: implement non_equivariant_dim for rotate comparison function
        raise NotImplementedError("Non equivariant dimension with rotation comparison function not implemented yet")

    if relational_encoding_type == "simple":
        raise NotImplementedError("SimpleRCNP with rotate comparison function is not implemented yet")
    elif relational_encoding_type == "full":
        if sparse and set_size > k:
            raise NotImplementedError("Sparse FullRCNP for rotation comparison function is not implemented yet")
        else:
            relational_matrix = B.sqrt(
                B.sum((xc.unsqueeze(2) - xc.unsqueeze(1)) ** 2, axis=-1).unsqueeze(
                    -1
                )
            )
            xt_matrix = B.sqrt(B.sum(xt.unsqueeze(2) ** 2, axis=-1)).unsqueeze(2).repeat(1, 1, set_size * set_size, 1)
            xc_matrix_1 = B.sqrt(B.sum(xc.unsqueeze(2) ** 2, axis=-1).unsqueeze(2)).repeat(1, 1, set_size, 1)
            xc_matrix_2 = B.sqrt(B.sum(xc.unsqueeze(1) ** 2, axis=-1).unsqueeze(-1)).repeat(1, set_size, 1, 1)
            yc_matrix_1 = yc.unsqueeze(2).repeat(1, 1, set_size, 1)
            yc_matrix_2 = yc.unsqueeze(1).repeat(1, set_size, 1, 1)
            # shape: [batch_size, set_size, set_size, 3]
            relational_matrix = B.concat(
                relational_matrix, xc_matrix_1, xc_matrix_2, yc_matrix_1, yc_matrix_2, axis=-1
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

            # shape: [batch_size, target_set_size, set_size * set_size, 4]
            dist_x = B.concat(
                dist_x,
                relational_matrix.unsqueeze(1).repeat(1, target_set_size, 1, 1),
                xt_matrix,
                axis=-1,
            )

            dist_x = dist_x.reshape(
                batch_size, target_set_size * set_size * set_size, -1
            )

            batch_size, encoding_size, filter_size = dist_x.shape
            # x shape: [batch_size * target_set_size * set_size * set_size, 4]
            encoding = dist_x.view(batch_size * encoding_size, filter_size)

    else:
        raise NotImplementedError

    return encoding, encoding_size
