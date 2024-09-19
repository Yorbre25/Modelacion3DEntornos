import torch

def grouping_operation(features, idx):
    """
    Agrupa características basadas en índices.

    :param features: Tensor de características con forma (B, C, N)
    :param idx: Tensor de índices con forma (B, npoint, nsample)
    :return: Tensor de características agrupadas con forma (B, C, npoint, nsample)
    """
    B, C, N = features.shape
    _, npoint, nsample = idx.shape
    
    # Expande 'idx' para que coincida con la forma de 'features'
    idx_expanded = idx.unsqueeze(1)  # Forma (B, 1, npoint, nsample)
    
    # Ajustar 'features' para usar 'torch.gather'
    features_expanded = features.unsqueeze(2).expand(-1, -1, npoint, -1)  # Forma (B, C, npoint, N)
    
    # Agrupa características usando los índices expandidos
    grouped_features = torch.gather(features_expanded, 3, idx_expanded)
    
    return grouped_features

# Datos de ejemplo
B, C, N = 1, 3, 100  # 1 batch, 3 características, 100 puntos
npoint, nsample = 10, 20  # 10 puntos de consulta, 20 vecinos
features = torch.rand(B, C, N)  # Tensor de características
idx = torch.randint(0, N, (B, npoint, nsample))  # Índices aleatorios

# Llamar a la función
grouped_features = grouping_operation(features, idx)

# Verificar la forma del tensor de salida
print(grouped_features.shape)  # Debería ser (B, C, npoint, nsample)




import matplotlib.pyplot as plt

grouped_features = grouping_operation(features, idx)

# Convertimos a numpy para graficar
grouped_features_np = grouped_features[0].numpy()  # Tomamos el primer lote

# Configuramos la visualización
fig, axes = plt.subplots(nrows=npoint, ncols=1, figsize=(10, 2 * npoint), sharex=True)

for i in range(npoint):
    ax = axes[i]
    # Graficamos las características de los puntos vecinos
    ax.plot(grouped_features_np[:, i, :].T)
    ax.set_title(f'Consulta Punto {i+1}')
    ax.set_ylabel('Características')
    ax.set_xlabel('Vecino')

plt.tight_layout()
plt.show()
