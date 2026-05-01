import matplotlib.pyplot as plt

# Define your color cycle here
color_cycle = ['#444344',  # Dunkelgrau
               '#9881FB',  # Violett
               '#2090CC',  # Blau
               '#59CC40',  # Grün
               '#B2B2B2',  # Hellgrau
               '#289792',  # Grünlich
               '#044859',  # Bläulich
               '#FB7140',  # Orange
               '#C53230']  # TU-ROT


def set_plot_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Palatino Linotype'  # Or use 'Palatino Linotype' if 'Palatino' is not available
    plt.rcParams['mathtext.fontset'] = 'stix'  # Keep using 'stix' for math text, as it resembles Palatino
    plt.rcParams['font.size'] = 12  # Adjust based on your document's specific needs
    plt.rcParams['figure.figsize'] = (10, 4)  # Adjust based on your document's layout and column width
    # plt.rcParams['text.usetex'] = True
    # Uncomment the above line if you want to enable LaTeX rendering for text in matplotlib
    # This will use your LaTeX installation to render text, including the Palatino font if it's available there.

    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_cycle)


def save_fig(file_name, **kwargs):
    # Set defaults if not provided
    save_args = {'bbox_inches': 'tight', 'pad_inches': 0}
    # Update with any additional keyword arguments provided
    save_args.update(kwargs)
    # Save the figure with the specified arguments
    # plt.savefig(file_name, **save_args)
    plt.tight_layout()
    plt.savefig(f'./plots/{file_name}.pdf', **save_args)
    plt.savefig(f'/home/laurenz/Documents/DAI/_Thesis/git/Thesis_template/Figures/plots/{file_name}.pdf', **save_args)
