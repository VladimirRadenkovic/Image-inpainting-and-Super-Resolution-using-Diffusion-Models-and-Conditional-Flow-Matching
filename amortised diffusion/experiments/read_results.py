#%%
import os
import yaml
import json

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
#%%
ci = 0.95
alpha = (1 - ci) / 2


def half_ci(group):
    data = group.dropna().to_numpy()
    sem = stats.sem(data)
    t2 = stats.t.ppf(1 - alpha, len(data) - 1) - stats.t.ppf(alpha, len(data) - 1)
    return sem * (t2 / 2)


def lower_ci(group):
    data = group.dropna().to_numpy()
    sem = stats.sem(data)
    mean = data.mean()
    t = stats.t.ppf(alpha, len(data) - 1)
    return mean + sem * t


def upper_ci(group):
    data = group.dropna().to_numpy()
    sem = stats.sem(data)
    mean = data.mean()
    t = stats.t.ppf(1 - alpha, len(data) - 1)
    return mean + sem * t


def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

from collections.abc import MutableMapping

def flatten(dictionary, parent_key='', separator='_'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def query(data_frame, query_string):
    if query_string == "all":
        return data_frame
    return data_frame.query(query_string)
# %%
root_path = "logs_eval"

configs = {}
results = {}
rows = []

# traverse root directory, and list directories as dirs and files as files
for root, dirs, files in walklevel(root_path, level=2):
    path = root.split(os.sep)
    try:
        if len(path) < 3:
            continue

        with open(os.path.join(root, "config.yaml"), 'r') as stream:
            config = yaml.safe_load(stream)
        with open(os.path.join(root,'results.json')) as json_data:
            result = json.load(json_data)
            json_data.close()
    except:
        continue
    # print('config', config)
    # print('results', result)
    # rows.append({**config, **result})
    rows.append({**flatten(config), **flatten(result), "path": root})

runs_df = pd.DataFrame(rows)
# %%

runs_df.shape
# %%

grouped_df = runs_df.groupby(
    ['dataset_name', 'conditioning_name']
)[['mse_mean', 'lpips_mean']].agg(['mean', stats.sem, "count"])
grouped_df

# %%

grouped_df.reset_index(inplace=True)

# %%
header = r"""
\begin{tabular}{lccc}
\toprule
\scshape Metric & \scshape Amortised & \scshape Guidance & \scshape Replacement \\"""

datasets = ["flowers", "celeba"]
methods = ["amortized", "reconstruction_guidance", "replacement",]
display_names = {
    "flowers": "Flowers",
    "celeba": "CelebA",
    "amortized": "Amortised",
    "reconstruction_guidance": "Guidance",
    "replacement": "Replacement",
    "mse_mean": r"MSE $(\downarrow)$",
    "lpips_mean": r"LPIPS $(\downarrow)$",
}

rows = []
for dataset in datasets:
    dataset_df = grouped_df[grouped_df['dataset_name'] == dataset]
    rows.append(r"\midrule")
    rows.append(r"\textbf{\scshape " + display_names[dataset] + r"} \\")
    for metric in ['mse_mean', 'lpips_mean']:
        row = [display_names[metric]]
        for conditioning in methods:
            df = dataset_df[dataset_df['conditioning_name'] == conditioning][metric]
            m = df['mean'].values[0]
            sem = df['sem'].values[0]
            row.append(f"${m:.2f}_{{\pm {sem:.2f}}}$")
        row[-1] += r"\\"
        rows.append(" & ".join(row))
rows.append(r"\bottomrule")
rows.append(r"\end{tabular}")
print(header)
print("\n".join(rows))
# %%
df = runs_df
df.columns
# %%

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

dataset = "flowers"
seed = 0
nrows = 8
IDX_TO_SAVE = 7
save = True
fig, axes = plt.subplots(nrows, 4, figsize=(5, 15))

for ax in axes.flatten():
    ax.axis('off')

for i in range(nrows):
    query = f"dataset_name == '{dataset}' and testing_seed == {seed} and conditioning_name == 'amortized'"
    gt_path = runs_df.query(query).iloc[0]['path']
    img = mpimg.imread(os.path.join(gt_path, "generated_groundtruth", f"image_gt_{10+i:03d}.jpg"))
    axes[i, 0].imshow(img)
    img = mpimg.imread(os.path.join(gt_path, "generated_groundtruth", f"image_gt2_{10+i:03d}.jpg"))
    axes[i, 0].imshow(img, alpha=0.3)

    if i == IDX_TO_SAVE and save:
        plt.figure(figsize=(5,5), frameon=False)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        img = mpimg.imread(os.path.join(gt_path, "generated_groundtruth", f"image_gt_{10+i:03d}.jpg"))
        plt.imshow(img)
        img = mpimg.imread(os.path.join(gt_path, "generated_groundtruth", f"image_gt2_{10+i:03d}.jpg"))
        plt.imshow(img, alpha=0.3)
        plt.axis('off')
        plt.savefig(f'plot_{dataset}_images/groundtruth.png')

    for j, method in enumerate(methods):
        query = f"dataset_name == '{dataset}' and testing_seed == {seed} and conditioning_name == '{method}'"
        gt_path = runs_df.query(query).iloc[0]['path']
        img = mpimg.imread(os.path.join(gt_path, "generated", f"image_{10+i:03d}.jpg"))
        axes[i, j+1].imshow(img)

        if i == IDX_TO_SAVE and save:
            plt.figure(figsize=(5,5), frameon=False)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(f'plot_{dataset}_images/{method}.png')

