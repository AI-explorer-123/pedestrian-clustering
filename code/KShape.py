from tqdm import tqdm, trange
import PIL
import io
import matplotlib.pyplot as plt
from tslearn.clustering import KShape
import pandas as pd
import numpy as np


def get_color(group_label):
    cmap = plt.get_cmap('tab10')
    color_index = group_label
    color = cmap(int(color_index))
    return color


n_clusters = 20

df = pd.read_csv('./data/students003.csv')
grouped = df.groupby('ID')

start_time = df['Time_step'].min()
end_time = df['Time_step'].max()

time_series_data = []
for name, group in grouped:
    group = group.set_index('Time_step').sort_index()
    full_traj = np.full((int(end_time - start_time + 1), 2), np.nan)
    for time, row in group.iterrows():
        full_traj[int(time - start_time)] = [row['X'], row['Y']]
    time_series_data.append(full_traj)

time_series_data = np.array(time_series_data)
mask = ~np.isnan(time_series_data)

model = KShape(n_clusters=n_clusters)
clusters = model.fit_predict(time_series_data)

time_steps = np.arange(start_time, end_time + 1)
people = {t: [] for t in time_steps}
for i, traj in enumerate(time_series_data):
    for t, coords in enumerate(traj):
        if not np.all(coords == 0):
            people[start_time +
                   t].append({'ID': i, 'loc': coords, 'label': clusters[i]})

n_frame = len(time_steps)
images = []

for t in trange(n_frame):
    time_step = time_steps[t]
    fig, ax = plt.subplots(1, figsize=(10, 10))
    for person in people[time_step]:
        color = get_color(person['label'])
        name = "Person" + str(int(person['ID']))
        ax.scatter(person['loc'][0], person['loc']
                   [1], linewidth=2, color=color)
        ax.text(person['loc'][0], person['loc'][1], name, color=color)
    ax.set_xlim([0, 15])
    ax.set_ylim([0, 15])
    ax.set_title(f"KShape - Time Step {time_step}")

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    image = PIL.Image.open(img_buf)
    images.append(image)
    plt.close()

images[0].save(
    "../media/KShape.gif",
    format="GIF",
    append_images=images[1:],
    save_all=True,
    duration=100,
    loop=0,
)
