import torch
from torch_geometric.data import Data

from gatr.interface import embed_vector
from experiments.unfolding.utils import get_pt
from experiments.unfolding.embedding import event_to_GA_with_spurions


class ZplusJetDataset(torch.utils.data.Dataset):
    def __init__(self, dtype, embed_into_GA=False, spurions=None):
        self.dtype = dtype
        self.embed_into_GA = embed_into_GA
        if embed_into_GA:
            self.spurions = spurions

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def create_data_list(
        self,
        det_particles,
        det_pids,
        det_mults,
        gen_particles,
        gen_pids,
        gen_mults,
    ):

        self.data_list = []
        for i in range(det_particles.shape[0]):
            det_pt = get_pt(det_particles[i])
            gen_pt = get_pt(gen_particles[i])
            det_idx = torch.argsort(det_pt, descending=True)[: det_mults[i]]
            gen_idx = torch.argsort(gen_pt, descending=True)[: gen_mults[i]]

            det_event = det_particles[i, det_idx]
            det_scalars = det_pids[i, det_idx]
            gen_event = gen_particles[i, gen_idx]
            gen_scalars = gen_pids[i, gen_idx]

            if self.embed_into_GA:
                if self.spurions is not None:
                    det_event, det_scalars = event_to_GA_with_spurions(
                        det_event, det_scalars, self.spurions
                    )
                else:
                    det_event = embed_vector(det_event).unsqueeze(-2)

            graph = Data(
                x_det=det_event,
                scalars_det=det_scalars,
                x_gen=gen_event,
                scalars_gen=gen_scalars,
            )

            self.data_list.append(graph)
