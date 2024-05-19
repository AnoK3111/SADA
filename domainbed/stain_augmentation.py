import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms as T


class stain_augmentation():
    def __init__(self, r=2, img_size=224, k=4, device='cpu',*args, **kwargs) -> None:
        self.r = r
        self.img_size = img_size
        self.k = k
        self.device = device
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.aug =  T.Compose(
            [
                T.ColorJitter(0.2, 0.25, 0.25, 0.15),
            ]
        )
    
    
    def SNMF(self, X, max_iter=10, lmbda=0.0001, eps=1e-6):
        H = torch.ones((X.shape[0], 2, X.shape[2])).float().to(self.device)
        B = torch.quantile(X, 0.01, dim=2, keepdim=True)
        W = torch.cat([torch.quantile(X, i/self.r, dim=2, keepdim=True) for i in range(1, self.r + 1)], dim=2)
        W = F.normalize(W, p=2, dim=1)
        V = X - B
        for i in range(max_iter):
            W *= ((V @ H.mT) / (W @ H @ H.mT + eps))
            W = F.normalize(W, p=2, dim=1)
            a = H / (W.mT @ W @ H + eps)
            H *= ((W.mT @ V) / (W.mT @ W @ H + eps))
            H = torch.maximum(H - a * lmbda, torch.tensor(0))

        return W, H, B
    
    def cluster(self, W, B, n_init=5, max_iters=100):
        X = torch.cat([W.reshape(W.size(0), self.r * 3), B.reshape(W.size(0), 3)], dim=1)
        time = 0
        best_loss = float('inf')
        best_centroids = None
        best_labels = None
        while time < n_init or best_centroids == None:
            centroids = X[torch.randperm(X.shape[0])[:self.k]]
            for _ in range(max_iters):
                distances = ((X - centroids[:, None])**2).sum(-1)
                labels = torch.argmin(distances, dim=0)
                new_centroids = torch.stack([X[labels == i].mean(0) for i in range(self.k)]) 
                if torch.all(centroids == new_centroids):
                    break
                centroids = new_centroids
            if torch.isnan(centroids).any():
                continue
            loss = ((X - centroids[labels])**2).sum()
            if loss < best_loss:
                best_loss = loss
                best_centroids = centroids
                best_labels = labels
            time += 1

        return best_labels, best_centroids.reshape(self.k, 3, self.r + 1)
    
    def domain_transfer(self, s, t, W, H, B, eps=1e-6):
        new_w = W[t]
        new_h = H[s] * ((torch.quantile(H[t], 0.99, dim=1) + eps) / (torch.quantile(H[s], 0.99, dim=1) + eps)).unsqueeze(-1)
        return (new_w @ new_h + B[t]).reshape(3, self.img_size, self.img_size)

    def transform(self, x, y):
        batch_size = x.shape[0]
        Imgs = self.convert_RGB_to_OD(x)
        W, H, B = self.SNMF(Imgs.reshape(batch_size, 3, -1))
        labels, centroid_W = self.cluster(W, B)
        new_x = torch.zeros((batch_size * self.k, 3, self.img_size, self.img_size)).float().to(self.device)
        new_y = torch.zeros((batch_size * self.k,)).long().to(self.device)
        index = 0
        choice = torch.arange(labels.shape[0]).to(self.device)
        for i in range(batch_size):
            new_x[index] = x[i]
            new_y[index] = y[i]
            index += 1 
            for j in range(self.k):
                if j == labels[i]:
                    continue
                s = i
                t = choice[labels == j]
                t = t[torch.randint(0, t.shape[0], [])]
                new_x[index] = self.domain_transfer(s, t, W, H, B)
                new_x[index] = self.aug(self.convert_OD_to_RGB(new_x[index]))
                new_y[index] = y[i]
                index += 1
        return self.norm(new_x), new_y
    
    def aug_only(self, x, y):
        batch_size = x.shape[0]
        new_x = torch.zeros((batch_size * self.k, 3, self.img_size, self.img_size)).float().to(self.device)
        new_y = torch.zeros((batch_size * self.k,)).long().to(self.device)
        index = 0
        for i in range(batch_size):
            new_x[index] = x[i]
            new_y[index] = y[i]
            index += 1 
            for j in range(1, self.k):
                new_x[index] = self.aug(x[i])
                new_y[index] = y[i]
                index += 1 
        return self.norm(new_x), new_y


    def origin_samples(self, x, y):
        return self.norm(self.aug(x)), y


    def copy_test(self, x, y):
        batch_size = x.shape[0]
        new_x = torch.zeros((batch_size * self.k, 3, self.img_size, self.img_size)).float().to(self.device)
        new_y = torch.zeros((batch_size * self.k,)).long().to(self.device)
        index = 0
        for i in range(batch_size):
            for _ in range(self.k):
                new_x[index] = x[i]
                new_y[index] = y[i]
                index += 1 
        return new_x, new_y

    
    def visuallize(self, x, w):
        x = self.transform(x, torch.zeros((x.size(0),)))[0]
        x = x * torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        x = x.cpu().numpy()
        x = np.rollaxis(x, 1, 4)
        batch_size = x.shape[0]
        plt.figure(figsize=(w, batch_size//w + 1))
        for i in range(batch_size):
            plt.subplot(batch_size//w + 1, w, i+1)
            plt.imshow(x[i])
        

    @staticmethod
    def convert_RGB_to_OD(I, eps=1e-6):
        I[((I[:,0,:,:] < 5e-2) & (I[:,1,:,:] < 5e-2) & (I[:,2,:,:] < 5e-2)).reshape(I.size(0), 1, 224, 224).repeat(1, 3, 1, 1)] = 1.0
        mask = (I < eps)
        I[mask] = eps
        return torch.maximum(-1 * torch.log(I), torch.tensor(eps)).float()

    @staticmethod
    def convert_OD_to_RGB(OD, eps=1e-6):
        assert OD.min() >= 0, "Negative optical density."
        OD = torch.maximum(OD, torch.tensor(eps))
        return (torch.exp(-1 * OD)).float()