from random import shuffle
from scipy.sparse import csr_matrix
import numpy as np
import scipy.sparse as sp
import torch
import time
import warnings

warnings.filterwarnings("ignore")
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")

class Data:
    def __init__(self, train_file, test_file, dataset_name, batch_size=256):
        self.train_file = train_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.Graph = None
        self.delimiter = "::"
        self.max_user_id = -1
        self.max_item_id = -1

        print("start loading data...")

        self.train_u = []
        self.train_i = []
        self.train_rating = []
        with open(self.train_file, 'r') as file:
            for line in file.readlines():
                parts = line.strip().split(self.delimiter)
                user_id = int(parts[0])
                item_id = int(parts[1])
                rating = float(parts[2])
                self.max_user_id = user_id if user_id > self.max_user_id else self.max_user_id
                self.max_item_id = item_id if item_id > self.max_item_id else self.max_item_id
                self.train_u.append(user_id - 1)
                self.train_i.append(item_id - 1)
                self.train_rating.append(rating)
        self.test_u = []
        self.test_i = []
        self.test_rating = []
        with open(self.test_file, 'r') as file:
            for line in file.readlines():
                parts = line.strip().split(self.delimiter)
                user_id = int(parts[0])
                item_id = int(parts[1])
                rating = float(parts[2])
                self.max_user_id = user_id if user_id > self.max_user_id else self.max_user_id
                self.max_item_id = item_id if item_id > self.max_item_id else self.max_item_id
                self.test_u.append(user_id - 1)
                self.test_i.append(item_id - 1)
                self.test_rating.append(rating)
        self.train_data = csr_matrix(
            (self.train_rating, (self.train_u, self.train_i)), shape=(
                self.max_user_id, self.max_item_id), dtype='float32')
        self.test_data = csr_matrix(
            (self.test_rating, (self.test_u, self.test_i)), shape=(
                self.max_user_id, self.max_item_id), dtype='float32')
        # (users,items), bipartite graph: sparse R, which is the adjacent matrix of UI
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.train_u)), (self.train_u, self.train_i)), shape=(
                self.max_user_id, self.max_item_id), dtype='float32')
        print("loading data done...")
        print("Number of users:", self.max_user_id, "Number of items:", self.max_item_id)
        # print(self.train_data)
        # print(self.UserItemNet)
        self.build_graph()

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def build_graph(self):
        print("loading adjacency matrix...")
        try:
            pre_adj_mat = sp.load_npz('pre_adj_mat/' + self.dataset_name + '_s_pre_adj_mat.npz')
            print("successfully loaded...")
            norm_adj = pre_adj_mat
        except:
            print("there is not be a generated adj_mat, generating adjacency matrix...")
            s = time.time()
            # adj_mat = sp.dok_matrix((self.max_user_id + self.max_item_id, self.max_user_id + self.max_item_id), dtype=np.float32)

            # adj_mat = adj_mat.tolil()
            # R = self.UserItemNet.tolil()
            # adj_mat[:self.max_user_id, self.max_user_id:] = R
            # adj_mat[self.max_user_id:, :self.max_user_id] = R.T
            # adj_mat = adj_mat.todok()
            # adj_coo = adj_mat.tocoo()

            R_coo = self.UserItemNet.tocoo()
            # print("R_coo_size: ", R_coo.shape)
            # print("adj_coo_size: ", adj_coo.shape)
            R_row = R_coo.row
            R_col = R_coo.col + self.max_user_id
            R_data = R_coo.data
            adj_row = np.hstack((R_col, R_row))
            adj_col = np.hstack((R_row, R_col))
            adj_data = np.hstack((R_data, R_data))
            adj_coo = sp.coo_matrix((adj_data, (adj_row, adj_col)))

            print("adj_coo_size: ", adj_coo.shape)
            adj_mat = adj_coo.todok()
            # print(adj_coo)
            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            end = time.time()
            print('costing ' + str(end - s) + 's, saved norm_mat...')
            # sp.save_npz('pre_adj_mat/' + self.dataset_name + '_s_pre_adj_mat.npz', norm_adj)
            print("generating adjacency matrix done...")

        self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        self.Graph = self.Graph.coalesce().to(device)

    def train_iterate_one_epoch(self, batch_size):
        user_temp = self.train_u[:]
        item_temp = self.train_i[:]
        labels_temp = self.train_rating[:]

        num_training = len(labels_temp)
        total_batch = int(num_training / batch_size)

        print("train_batch", num_training, "total_batch: ", total_batch)

        idxs = np.random.permutation(num_training)  # shuffled ordering
        user_random = list(np.array(user_temp)[idxs])
        item_random = list(np.array(item_temp)[idxs])
        labels_random = list(np.array(labels_temp)[idxs])

        return user_random, item_random, labels_random, total_batch

    def test_iterate_one_epoch(self, batch_size):
        user_temp = self.test_u[:]
        item_temp = self.test_i[:]
        labels_temp = self.test_rating[:]

        num_testing = len(labels_temp)
        total_batch = int(num_testing / batch_size)

        print("test_batch: ", num_testing, "total_batch: ", total_batch)

        idxs = np.random.permutation(num_testing)  # shuffled ordering
        user_random = list(np.array(user_temp)[idxs])
        item_random = list(np.array(item_temp)[idxs])
        labels_random = list(np.array(labels_temp)[idxs])

        return user_random, item_random, labels_random, total_batch, num_testing
