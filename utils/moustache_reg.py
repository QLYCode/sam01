import torch
import torch.nn as nn
import math

class MoustacheRegularizer(nn.Module):
    super()
    def __init__(self, num_clusters=4, embedding_size=2048, learning_rate=0.1, lr_decay=0.99, running_avg_factor=0.01, device="cuda"):
        super(MoustacheRegularizer, self).__init__()
        self.num_clusters = num_clusters
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.device = device
        self.centroids = torch.randn(self.num_clusters, self.embedding_size).to(self.device)
        self.running_avg_factor=running_avg_factor
        self.running_avg = 0
        self.running_sd = 1

    def fit(self, new_data):
        distances = self._compute_distances(new_data) 
        labels = torch.argmin(distances, dim=1)
        for j in range(self.num_clusters):
            cluster_points = new_data[labels == j]
            if len(cluster_points) > 0:
                mean_point = cluster_points.mean(dim=0)
                self.centroids[j] = self.centroids[j] + self.learning_rate * (mean_point - self.centroids[j])
        self.learning_rate *= self.lr_decay


    def labels(self, data):
        distances = self._compute_distances(data)
        return torch.argmin(distances, dim=1)
    
    def _get_normalized(self, distances):
        min_dist = torch.min(distances, dim=1).values
        if self.running_avg is None:
            self.running_avg = torch.mean(min_dist).item()
            self.running_sd = torch.std(min_dist).item()
        else:
            self.running_avg = (1 - self.running_avg_factor) * self.running_avg + self.running_avg_factor * torch.mean(min_dist).item()

            self.running_sd = (1 - self.running_avg_factor) * self.running_sd + self.running_avg_factor * torch.std(min_dist).item()

        if self.running_sd > 0:
            normalized_min_dist = (min_dist - self.running_avg) / self.running_sd
        else:
            normalized_min_dist = torch.zeros_like(min_dist)
        return normalized_min_dist

    def get_deltas(self, data):
        with torch.no_grad():
            data -= self.running_avg
            data /= self.running_sd
            c=0.1
            result = -(torch.square(data) + torch.cos(data * math.pi))+0.9
            result = torch.clamp(result * c, min=-c, max=c)
            result = 1 + result
        return result

    def forward(self, data):
        distances = self._compute_distances(data)
        distances = self._get_normalized(distances)
        return distances


    def _compute_distances(self, data):
        return torch.norm(data[:, None, :] - self.centroids[None, :, :], dim=2)

if __name__ == "__main__":
    torch.manual_seed(42)
    data = torch.normal(mean=5, std=2, size=(100, 10))
    kmeans = MoustacheRegularizer()
    kmeans.fit(data)
    labels = kmeans.labels(data)
    print(f"Cluster Labels: {labels}")
    new_sample = torch.normal(mean=5, std=2, size=(1, 10))
    distance = kmeans.transform(new_sample)
    print(f"Distance to the closest cluster center: {distance.item():.2f}")