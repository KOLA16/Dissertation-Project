import torch
import torch.nn.functional as F


class AaD:

    def __init__(self, k, alpha, beta, tar_size):
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.tar_size = tar_size
        self.feature_bank = None
        self.score_bank = None

        self.true_labels_bank = None

    def init_memory_banks_step(self, indices, f, softmax_output, y):
        with torch.no_grad():
            # check if memory banks have been initialised
            if self.feature_bank is None and self.score_bank is None and self.true_labels_bank is None:
                self.feature_bank = torch.randn(self.tar_size, f.shape[1]).cuda()
                self.score_bank = torch.randn(self.tar_size, softmax_output.shape[1]).cuda()
                self.true_labels_bank = torch.randn(self.tar_size, 1)

                f_norm = F.normalize(f)
                self.feature_bank[indices] = f_norm.detach().clone()
                self.score_bank[indices] = softmax_output.detach().clone()
                self.true_labels_bank[indices] = y.detach().clone().cpu()
            else:
                f_norm = F.normalize(f)
                self.feature_bank[indices] = f_norm.detach().clone()
                self.score_bank[indices] = softmax_output.detach().clone()
                self.true_labels_bank[indices] = y.detach().clone().cpu()

    def adaptation_step(self, indices, f, softmax_output):
        with torch.no_grad():
            f_norm = F.normalize(f)
            self.feature_bank[indices] = f_norm.detach().clone()
            self.score_bank[indices] = softmax_output.detach().clone()
            near_softmax_output = self._find_nearest_neighbours_score(f_norm)

        loss, first_term, second_term = self._loss(softmax_output, near_softmax_output)

        return loss, first_term, second_term

    def _find_nearest_neighbours_score(self, f):
        f_ = f.detach().clone()
        cos_sim = f_ @ self.feature_bank.T
        _, near_indices = torch.topk(cos_sim, dim=-1, largest=True, k=self.k+1)
        near_indices = near_indices[:, 1:]
        near_softmax_output = self.score_bank[near_indices]

        return near_softmax_output

    def _loss(self, softmax_output, near_softmax_output):
        softmax_output_un = softmax_output.unsqueeze(1).expand(-1, self.k, -1)

        # equivalent to mean of a dot product between softmax predictions for samples in the batch
        # and softmax predictions of their nearest neighbours
        # objective is to maximise mean value of this dot product which corresponds to achieving more similar
        # predictions between the nearest neighbours (prediction consistency)
        loss = -torch.mean(torch.einsum('ijk,ijk->ij', softmax_output_un, near_softmax_output).sum(-1))
        first_term = loss.item()

        mask = torch.ones(softmax_output.shape[0], softmax_output.shape[0])  # batch x batch
        mask.fill_diagonal_(0)  # matrix of ones with zeros on a diagonal
        copy = softmax_output.T
        dot_neg = softmax_output @ copy  # dot product of softmax output with itself transposed
                                         # i.e. the second term of the loss equation done on all
                                         # features from the batch at once and before summation applied.
                                         # Cells in the dot_neg matrix are all possible
                                         # dot products between predictions for each feature in a batch
                                         # and all the remaining features in a batch. Diagonal cells are
                                         # dot products of predictions for the same feature.
        dot_neg = (dot_neg * mask.cuda()).sum(-1)  # summation of dot products, and we ignore
                                                   # elements on the diagonal, hence multiplication by the mask
        neg_pred = torch.mean(dot_neg)
        second_term = neg_pred.item()

        loss += neg_pred * self.alpha

        return loss, first_term, second_term
