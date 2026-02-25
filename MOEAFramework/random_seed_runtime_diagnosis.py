import os
import numpy as np
import pandas as pd
import matplotlib
from pathlib import Path
#matplotlib.use('TkAgg')  # Switch to a more standard backend
import matplotlib.pyplot as plt
# import pathnavigator
 
# pn = pathnavigator.create(os.path.dirname(__file__))
# pn.chdir()

 
#%% Plot random seed runtime diagonsis (hypervolume)

def read_moea_metrics_file(file_path):
    with open(file_path, 'r') as file:
        # Read the first line to get the column names and replace '#' with ''
        columns = [col.replace('#', '') for col in file.readline().strip().split()]

        # Read the rest of the file into a DataFrame
        df = pd.read_csv(file, sep='\\s+', names=columns)

    return df

dir = "/Users/keaniw/Documents/Research/Chile Project/MOEAFramework"
#folder = "DPS_results_nFunc200000/SRI12_2020_2040_CMIP6_04Feb25"
fol = "DPS_results_Validation_combined"
folders = [f for f in Path(f'{dir}/{fol}').iterdir() if f.is_dir()]
freq = 100  # Borg runtime output frequency

for folder in folders:

    #files = [i for i in os.listdir(pn.get() / folder) if ".metrics" in i]
    files = [i for i in os.listdir(dir + "/" + fol + "/" + folder.name) if ".metrics" in i]

    df = pd.DataFrame()
    for file in files:
        #file_path = pn.get() / folder / file
        file_path = dir + "/" + fol + "/" + folder.name + "/" + file
        # seed = [int(i[4:]) for i in file.split(".")[0].split("_") if "seed" in i][0]
        seed = [int(i[3]) for i in [file.split(".")[0].split("_")] if "seed" in i][0]
        df[seed] = read_moea_metrics_file(file_path).Hypervolume

    plt.figure(figsize=(10, 3))
    fig, ax = plt.subplots()
    df.plot(ax=ax, legend=False)
    ax.set_xlabel("NFE")
    ax.set_ylabel("Hypervolume")
    #xticks = df.index[::3]
    xticks = df.index[::2]
    ax.set_xticks(xticks)  # Set tick positions
    #ax.set_xticklabels([str(np.array(i+1) * freq) if i % 2 == 1 else "" for i in range(len(xticks))])  # Label only even indices
    ax.set_xticklabels(np.array(xticks+1) * freq)  # Set tick labels as xticks * freq
    #xmax = 19  # max xlim is xmax*freq
    xmax = 40
    #ax.set_xlim([0, xmax])
    ax.set_xlim([-.5, xmax])
    #plt.xlim([0, xmax])
    ax.legend(ncols=5, frameon=False, title="Seed")
    #plt.title('SRI12 2020-2040')
    plt.title(f'{folder.name}')
    plt.gcf().set_size_inches(6, 3)
    plt.tight_layout()
    plt.show()

#%% Plot interactive parallel axes using plotly

import plotly.graph_objects as go

def plotly_parallel_plot(df, reference_values, hue='Obj1', orders=None, browser=True):
    if browser:
        import plotly.io as pio
        pio.renderers.default = "browser"

    # Normalize reference values to match the Parcoords scale (0-1 range)
    def normalize(value, min_val, max_val):
        return (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5

    # Compute the range for each dimension
    options = {
        k: dict(label=k, values=df[k], range=[0, 1])
        for k in df
    }

    if orders is None:
        orders = list(options.keys())
    dimensions = [options[i] for i in orders]

    # Normalize reference values according to the axis ranges
    if reference_values is not None:
        normalized_ref_values = [
            normalize(reference_values[dim["label"]], dim["range"][0], dim["range"][1]) for dim in dimensions
        ]

    # Create Parallel Coordinates Figure
    fig = go.Figure()

    fig.add_trace(
        go.Parcoords(
            line=dict(color=df[hue],
                      colorscale='Tealrose',
                      showscale=True,
                      colorbar=dict(
                          title=hue,  # Set color bar label
                      )
                      ),
            dimensions=dimensions
        )
    )

    if reference_values is not None:
        # Add Overlay Line
        fig.add_trace(go.Scatter(
            x=[dim["label"] for dim in dimensions],
            y=normalized_ref_values,
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=8),
            name='Overlay Line'
        ))

        # Force Y-axis limits from 0 to 1
        fig.update_layout(
            xaxis=dict(
                showticklabels=False,  # Hide x-axis labels
                showgrid=False,  # Remove x-grid
                zeroline=False,  # Remove x-axis zero line
                range=[0, len(orders) - 1]
            ),
            yaxis=dict(
                showticklabels=False,  # Hide y-axis labels
                showgrid=False,  # Remove y-grid
                zeroline=False,  # Remove y-axis zero line
                range=[0, 1]  # Keep y in 0-1 range
            ),
        )

    fig.show()


#% Process ref
dir = "/Users/keaniw/Documents/Research/Chile Project/MOEAFramework"
#folder = "DPS_results_nFunc200000/SRI12_2020_2040_CMIP6_04Feb25"
#fol = "DPS_results_Validation_combined"
fol = "DPS_results_Validation_combined_disc0"
# fol = "DPS_results_nFunc200000"
# fol = "DPS_results_nFunc2000"
folders = [f for f in Path(f'{dir}/{fol}').iterdir() if f.is_dir()]

for folder in folders:

    # file = pn.get() / folder / "borg.ref"
    file = f'{dir}/{fol}/{folder.name}/borg.ref'

    df = pd.read_csv(file, sep='\\s+', header=None)
    #df[2] = -1 * df[2]

    # Output csv for J3
    obj_names = [f"Obj{i + 1}" for i in range(df.shape[1])]
    df.columns = obj_names

    #% Plot interactive parallel axises using plotly
    reference_values = {k: 0 for k in obj_names}

    plotly_parallel_plot(df, reference_values=None, hue='Obj1',
                         orders=obj_names, browser=True)


#%% Do not constrain axes bounds

import plotly.graph_objects as go
def plotly_parallel_plot(df, title=None, reference_values=None, hue='Obj1', orders=None, save_path=None):
    if orders is None:
        orders = list(df.columns)

    # Use data-driven ranges for each objective
    dimensions = []
    for col in orders:
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        dimensions.append(
            dict(
                label=col,
                values=df[col],
                range=[col_min, col_max],
            )
        )

    fig = go.Figure()

    fig.add_trace(
        go.Parcoords(
            line=dict(
                color=df[hue],
                colorscale='Tealrose',
                showscale=True,
                colorbar=dict(title=hue),
            ),
            dimensions=dimensions
        )
    )

    # OPTIONAL: overlay reference line (in original units)
    if reference_values is not None:
        fig.add_trace(
            go.Scatter(
                x=[d["label"] for d in dimensions],
                y=[reference_values.get(d["label"], None) for d in dimensions],
                mode='lines+markers',
                line=dict(color='red', width=3),
                marker=dict(size=8),
                name='Reference'
            )
        )

    # Set figure title
    if title is not None:
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,  # center title
                xanchor='center'
            )
        )

    # --- Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(save_path))

    fig.show()


#% Process ref
dir = "/Users/keaniw/Documents/Research/Chile Project/MOEAFramework"
#folder = "DPS_results_nFunc200000/SRI12_2020_2040_CMIP6_04Feb25"
fol = "DPS_results_Validation_combined"
fol = "DPS_results_Validation"
fol = "DPS_results_nFunc2000"
fol = "DPS_results_Validation_combined_disc0"
# fol = "DPS_results_nFunc200000"
# fol = "DPS_results_nFunc2000"
folders = [f.name for f in Path(f'{dir}/{fol}').iterdir() if f.is_dir()]

for folder in folders:

    # file = pn.get() / folder / "borg.ref"
    file = f'{dir}/{fol}/{folder}/borg.ref'
    fig_save_path = f'{dir}/plots/{folder}_parallel.png'

    df = pd.read_csv(file, sep='\\s+', header=None)
    df[2] = -1 * df[2] / 1E6  # convert units

    # Output csv for J3
    obj_names = [f"Obj{i + 1}" for i in range(df.shape[1])]
    obj_names = ['Urb. Rel.', 'Ag. Rel.', 'Total Cost']
    df.columns = obj_names

    #% Plot interactive parallel axises using plotly
    reference_values = {k: 0 for k in obj_names}

    plotly_parallel_plot(df, title=folder, reference_values=None, hue=obj_names[0],
                         orders=obj_names, save_path=fig_save_path)
