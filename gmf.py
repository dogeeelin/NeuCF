import torch
from engine import Engine
from utils import use_cuda


class GMF(torch.nn.Module):
    def __init__(self, config):
        super(GMF, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']


        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1) # 8 -> 1
        self.logistic = torch.nn.Sigmoid()


        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip([16, 64, 32, 16], [64, 32, 16, 8])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))


    def forward(self, user_indices, item_indices):
        
        # user_indices [1024]
        # item_indices [1024]
        # element_product [1024, 8]
        # rating [1024,1]

        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)

        # # element_product
        # element_product = torch.mul(user_embedding, item_embedding) # shape: [1024, 8]
        # logits = self.affine_output(element_product)  # 8 -> 1  
        # rating = self.logistic(logits)              # 1 -> 1


        # inner_product
        inner_product = torch.sum(user_embedding*item_embedding, dim=1)
        rating = self.logistic(inner_product)

        # # L1 distance
        # L1 = torch.abs(user_embedding - item_embedding).sum(dim=1)
        # rating = self.logistic(L1)

        # # L2 distance 
        # L2 = torch.norm(user_embedding.float() - item_embedding.float(), dim=1)
        # rating = self.logistic(L2)


        # # MLP
        # vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        # for idx, _ in enumerate(range(len(self.fc_layers))):
        #     vector = self.fc_layers[idx](vector)
        #     vector = torch.nn.ReLU()(vector)
        # logits = self.affine_output(vector)
        # rating = self.logistic(logits)


        return rating # rating [1024,1]

    def init_weight(self):
        pass


class GMFEngine(Engine): # 这里面使用了上面的 GMF 的模型，并用 CUDA 训练
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = GMF(config) # 实例化 GMF 模型
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(GMFEngine, self).__init__(config)