import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def setup_thesis_style():
    """
    Configura estilo profesional consistente para todas las figuras.
    """
    # Estilo base
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Parámetros personalizados
    thesis_params = {
        # Fuentes
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.title_fontsize': 10,
        
        # Líneas y markers
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        
        # Ejes
        'axes.linewidth': 0.8,
        'axes.grid': False,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        
        # Figura
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # LaTeX
        'text.usetex': False,  # True si tienes LaTeX instalado
        'mathtext.fontset': 'stix',
    }
    
    mpl.rcParams.update(thesis_params)
    
    return thesis_params

# Paleta de colores profesional (colorblind-friendly, print-safe)
COLORS = {
    # Primarios
    'train': '#0077BB',       # Azul fuerte - entrenamiento
    'val': '#CC3311',         # Rojo - validación
    
    # Parámetros
    'D': '#0077BB',           # Azul - DMI
    'sigma': '#EE7733',       # Naranja - sigma
    
    # Componentes de loss
    'regression': '#0077BB',  # Azul
    'spectral': '#009988',    # Verde azulado
    'energy': '#EE7733',      # Naranja
    
    # Referencias
    'reference': '#555555',   # Gris oscuro
    'highlight': '#CC3311',   # Rojo para destacar
    'neutral': '#BBBBBB',     # Gris claro
}


# ==============================================================================
# CONFIGURACIÓN DE ESTILO (usa el que ya definiste)
# ==============================================================================
setup_thesis_style()
# ==============================================================================
# ASUME QUE setup_thesis_style() Y COLORS YA FUERON DEFINIDOS
# ==============================================================================
setup_thesis_style()

def plot_D_sigma_heatmap(
    csv_path,
    D_col="Dind",
    sigma_col="sigma_nominal",
    bins_D=40,
    bins_sigma=40,
    log_scale=False,
    D_min=0.3,      # mJ/m^2
    D_max=1.5,      # mJ/m^2
    sigma_min=0.02, # J/m^2
    sigma_max=0.15, # J/m^2
    save_path=None
):
    """
    Mapa de calor del conteo de muestras en el plano (D, sigma),
    restringido al dominio físico de interés.
    """

    # ---------------------------
    # Leer datos
    # ---------------------------
    df = pd.read_csv(csv_path)

    if D_col not in df.columns or sigma_col not in df.columns:
        raise ValueError("Las columnas especificadas no existen en el CSV")

    # ---------------------------
    # Conversión de unidades
    # ---------------------------
    # D: J/m^2 → mJ/m^2
    D_mJ = 1e3 * df[D_col].values
    sigma = df[sigma_col].values  # ya está en J/m^2

    # ---------------------------
    # Filtro del dominio físico
    # ---------------------------
    mask = (
        (D_mJ >= D_min) & (D_mJ <= D_max) &
        (sigma >= sigma_min) & (sigma <= sigma_max)
    )

    D_mJ = D_mJ[mask]
    sigma = sigma[mask]

    # ---------------------------
    # Binning y conteo
    # ---------------------------
    H, D_edges, sigma_edges = np.histogram2d(
        D_mJ,
        sigma,
        bins=[bins_D, bins_sigma],
        range=[[D_min, D_max], [sigma_min, sigma_max]]
    )

    if log_scale:
        H = np.log10(H + 1.0)

    # ---------------------------
    # Plot
    # ---------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    im = ax.imshow(
        H.T,
        origin="lower",
        aspect="auto",
        extent=[D_min, D_max, sigma_min, sigma_max],
        cmap="viridis"
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(
        r"$\log_{10}(\mathrm{conteo}+1)$" if log_scale else "Conteo de muestras"
    )

    # Etiquetas con unidades
    ax.set_xlabel(r"Constante DMI $D$ (mJ/m$^2$)")
    ax.set_ylabel(r"Dispersión de anisotropía $\sigma$ (J/m$^2$)")

    # Título científico
    ax.set_title(
        "Cobertura del espacio de parámetros del dataset en el plano $(D,\,\sigma)$"
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    #plt.show()


# ==============================================================================
# EJEMPLO DE USO
# ==============================================================================
plot_D_sigma_heatmap(
    csv_path="../../../../mumax_dataset_ku_by_block_disorder_phy_corrected_3/training_index.csv",
    bins_D=40,
    bins_sigma=40,
    log_scale=False,
    save_path="D_sigma_coverage_heatmap.pdf"
)

