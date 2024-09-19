import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Propósito:
    Encuentra los puntos en un conjunto de puntos 3D que están dentro de una esfera de radio especificado alrededor de puntos de consulta.

Entrada:
    -radius: El radio de la esfera alrededor del punto de consulta.
    -nsample: El número máximo de puntos que se desean muestrear dentro de la esfera.
    -pts: Conjunto de puntos en 3D (generalmente un tensor o matriz con forma [B, N, 3], 
    donde B es el número de lotes, N es el número de puntos por lote, y 3 representa las coordenadas x, y, z).
    -new_pts: Puntos de consulta en 3D (generalmente un tensor o matriz con forma [B, S, 3], 
    donde S es el número de puntos de consulta por lote).

Salida:
    Un tensor que contiene los índices de los puntos que están dentro del radio para cada punto de consulta.
    La forma del tensor suele ser [B, S, K], donde K es el número de puntos encontrados dentro del radio 
    (que puede variar para diferentes puntos de consulta), o bien K puede ser limitado por nsample
"""


def ball_query(pts, new_pts, radius, nsample):
    """
    :param pts: all points, [B, N, 3]
    :param new_pts: query points, [B, S, 3]
    :param radius: local spherical radius
    :param nsample: max sample number in local sphere
    :return: indices of sampled points around new_pts [B, S, nsample]
    """
    device = pts.device
    B, N, C = pts.shape
    _, S, _ = new_pts.shape

    # Create an empty tensor to hold the indices of sampled points
    sampled_indices = torch.zeros(B, S, nsample, dtype=torch.long, device=device)

    for b in range(B):
        # Calculate pairwise distances between all points and query points
        pts_b = pts[b]  # [N, 3]
        new_pts_b = new_pts[b]  # [S, 3]
        
        # Expand dimensions for broadcasting
        pts_exp = pts_b.unsqueeze(0)  # [1, N, 3]
        new_pts_exp = new_pts_b.unsqueeze(1)  # [S, 1, 3]

        # Compute squared distances
        dists_sq = torch.sum((pts_exp - new_pts_exp) ** 2, dim=-1)  # [S, N]

        # Find points within the radius
        mask = dists_sq <= radius ** 2
        for s in range(S):
            indices = torch.nonzero(mask[s]).squeeze(1)  # [num_points_in_sphere]

            if indices.numel() > nsample:
                # If there are more points than nsample, randomly sample
                indices = indices[torch.randperm(indices.size(0))[:nsample]]
            elif indices.numel() < nsample:
                # If there are fewer points than nsample, pad with zeros if indices is empty
                pad_size = nsample - indices.numel()
                if pad_size > 0:
                    if indices.numel() == 0:
                        # No points found, pad with zeros (or any other placeholder index like -1)
                        indices = torch.zeros(nsample, dtype=torch.long, device=device)
                    else:
                        # Repeat the last index to fill remaining spots
                        indices = torch.cat([indices, indices[-1].repeat(pad_size)])
            
            # Store the indices of the sampled points
            sampled_indices[b, s, :indices.numel()] = indices

    return sampled_indices


# Example usage
torch.manual_seed(42)
pts = torch.rand(2, 100, 3)  # Example points [B, N, 3]
torch.manual_seed(42)
new_pts = torch.rand(2, 10, 3)  # Example query points [B, S, 3]
radius = 0.1
nsample = 5

sampled_pts = ball_query(pts, new_pts, radius, nsample)
print(sampled_pts.shape)

def plot_ball_query(pts, new_pts, indices, radius):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for b in range(pts.shape[0]):
        pts_b = pts[b].cpu().numpy()  # [N, 3]
        new_pts_b = new_pts[b].cpu().numpy()  # [S, 3]
        indices_b = indices[b].cpu().numpy()  # [S, nsample]

        # Plot all points
        ax.scatter(pts_b[:, 0], pts_b[:, 1], pts_b[:, 2], c='b', marker='o', label='All Points')

        # Plot query points
        ax.scatter(new_pts_b[:, 0], new_pts_b[:, 1], new_pts_b[:, 2], c='r', marker='^', s=100, label='Query Points')

        # For each query point, plot the sampled points within the sphere
        for s in range(new_pts_b.shape[0]):
            sampled_pts = pts_b[indices_b[s]]  # [nsample, 3]
            
            # Plot the sampled points
            ax.scatter(sampled_pts[:, 0], sampled_pts[:, 1], sampled_pts[:, 2], c='g', marker='x', s=60, label='Sampled Points' if s == 0 else None)

            # Draw a sphere around the query point
            u, v = torch.meshgrid(torch.linspace(0, 2 * torch.pi, 20), torch.linspace(0, torch.pi, 20))
            x = radius * torch.cos(u) * torch.sin(v) + new_pts_b[s, 0]
            y = radius * torch.sin(u) * torch.sin(v) + new_pts_b[s, 1]
            z = radius * torch.cos(v) + new_pts_b[s, 2]
            ax.plot_wireframe(x.numpy(), y.numpy(), z.numpy(), color="r", alpha=0.2)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Ball Query Results')
    plt.legend()
    plt.show()

# Plot the results
plot_ball_query(pts, new_pts, sampled_pts, radius)

