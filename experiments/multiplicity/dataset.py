import torch


class MultiplicityDataset(torch.utils.data.Dataset):
    def __init__(self, particles, multiplicities, dtype):
        self.particles = [
            torch.tensor(particles_onedataset, dtype=dtype)
            for particles_onedataset in particles
        ]
        self.multiplicities = [
            torch.tensor(multiplicities_onedataset, dtype=dtype)
            for multiplicities_onedataset in multiplicities
        ]

        # reduce the effectively used dataset to the length of the smallest dataset
        # (pure convenience, could use more data at the cost of more code)
        self.len = min(
            [len(particles_onedataset) for particles_onedataset in self.particles]
        )

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return [
            (particles[idx], multiplicities[idx])
            for (particles, multiplicities) in zip(self.particles, self.multiplicities)
        ]
