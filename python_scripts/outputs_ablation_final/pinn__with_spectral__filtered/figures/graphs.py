import pandas as pd
import matplotlib.pyplot as plt



# ==============================================================================
# ESTILO UNIFICADO PARA FIGURAS DEL TRABAJO DE GRADO
# ==============================================================================

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
        'axes.grid': True,
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

# Llamar al inicio de tu notebook
setup_thesis_style()

# ==============================================================================
# CONFIGURACIÓN DE ESTILO (usa el que ya definiste)
# ==============================================================================
setup_thesis_style()

# ==============================================================================
# FUNCIÓN DE PLOT
# ==============================================================================
def plot_physical_annealing(
    csv_path,
    epoch_col="epoch",
    weight_col="weight_spectral",
    save_path=None
):
    """
    Grafica la evolución del peso físico (annealing) durante el entrenamiento PINN.
    """

    # Cargar datos
    df = pd.read_csv(csv_path)

    # Ordenar por época por seguridad
    df = df.sort_values(epoch_col)

    # Crear figura
    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    # Plot principal
    ax.plot(
        df[epoch_col],
        df[weight_col],
        color=COLORS['spectral'],
        label=r"$\lambda_{\mathrm{spectral}}$",
    )

    # Etiquetas
    ax.set_xlabel("Época")
    ax.set_ylabel("Peso del término espectral")

    # Límites opcionales (si sabes que va de 0 a 1, por ejemplo)
    # ax.set_ylim(0, 1.05)

    # Leyenda
    ax.legend(frameon=True)

    # Ajustes finales
    ax.set_title("Evolución del peso espectral durante el entrenamiento de la PINN")

    plt.tight_layout()

    # Guardar si se solicita
    if save_path is not None:
        plt.savefig(save_path)

    #plt.show()


# ==============================================================================
# EJEMPLO DE USO
# ==============================================================================
plot_physical_annealing(
    csv_path="training_history_pinn__with_spectral__filtered.csv",
    save_path="annealing_spectral_weight.png"
)

