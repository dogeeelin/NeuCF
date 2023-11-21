import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

random.seed(0)


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class SampleGenerator_gowalla(object):
    """Construct dataset for NCF"""

    def __init__(self, ratings, train_data, test_data):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        self.ratings = ratings
        # explicit feedback using _normalize and implicit using _binarize
        # self.preprocess_ratings = self._normalize(ratings)
        self.preprocess_ratings = self._binarize(ratings) # 二值化 0代表没有互动，1代表有互动。
        self.user_pool = set(self.ratings['userId'].unique()) # 6040 有哪些
        self.item_pool = set(self.ratings['itemId'].unique()) # 3706
        # create negative item samples for NCF learning
        self.negatives = self._sample_negative(ratings) # Index(['userId', 'negative_items', 'negative_samples'], dtype='object')
        self.train_ratings, self.test_ratings = train_data, test_data

    def _normalize(self, ratings):
        """normalize into [0, 1] from [0, max_rating], explicit feedback"""
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating * 1.0 / max_rating
        return ratings
    
    def _binarize(self, ratings):
        """binarize into 0 or 1, imlicit feedback"""
        ratings = deepcopy(ratings)
        ratings['rating'][ratings['rating'] > 0] = 1.0 # 0代表没有互动，1代表有互动。
        return ratings

    # def _split_loo(self, ratings):
    #     """leave one out train/test split """
    #     # 这段代码将为这个数据框创建一个新的列 rank_latest。该列将包含每个用户评分的时间戳的排名，每个用户都有自己对一系列物品评分的时间排名，把排名都是 1 的抽出来。越晚产生的数据，排名约小
    #     ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
    #     test = ratings[ratings['rank_latest'] == 1]
    #     train = ratings[ratings['rank_latest'] > 1]
    #     assert train['userId'].nunique() == test['userId'].nunique()
    #     return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def _sample_negative(self, ratings):
        """return all negative items & 100 sampled negative items"""
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x) # 挑出没有交集的 items
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(sorted(x), 99)) # 随机挑选99个无交集的 items
        return interact_status[['userId', 'negative_items', 'negative_samples']]
        # negative_items 是所有的无交集的物品，negative_samples 是随机挑选的 99 个无交集的样本

    def instance_a_train_loader(self, num_negatives, batch_size): # loader对数据按照 batch_size 256进行打包，包括用户，物品，评分，都是 256 个。
        """instance train loader for one training epoch"""
        users, items, ratings = [], [], []
        train_ratings = pd.merge(self.train_ratings, self.negatives[['userId', 'negative_items']], on='userId') # 合并两个df
        train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(sorted(x), num_negatives))  # 随机选出 4 个无交集的物品
        for row in train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))
            for i in range(num_negatives):
                users.append(int(row.userId))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))  # negative samples get 0 rating
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
# DataLoader，它是PyTorch中数据读取的一个重要接口，该接口的目的：将自定义的Dataset根据batch size大小、是否shuffle等封装成一个Batch Size大小的Tensor，用于后面的训练。

    @property
    def evaluate_data(self):
        """create evaluate data"""
        test_ratings = pd.merge(self.test_ratings, self.negatives[['userId', 'negative_samples']], on='userId')
        test_users, test_items, negative_users, negative_items = [], [], [], []
        for row in test_ratings.itertuples():
            test_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            for i in range(len(row.negative_samples)):
                negative_users.append(int(row.userId))
                negative_items.append(int(row.negative_samples[i]))
        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
                torch.LongTensor(negative_items)]
