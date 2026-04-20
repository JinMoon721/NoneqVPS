import torch


def make_bias_force_fn(model, device="cpu"):
    def bias_force_fn(atoms, tau):
        """
        atoms.positions: (N, 3)
        tau (1)
        
        Returns
            doob force (N, 3)
        """
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float32).view(1, 1, -1, 3)
        cell = torch.tensor(atoms.cell.array, dtype=torch.float32).view(1, 1, 3, 3)
        tau_t = torch.tensor([[tau]], dtype=torch.float32)
        
        with torch.enable_grad():
            _, _, doob = model.doob(pos, tau_t, cell, create_graph=False)
        
        return doob[0,0].detach().cpu().numpy()
    return bias_force_fn


