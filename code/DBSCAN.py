from tqdm import tqdm, trange
from model import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import PIL


def get_color(group_label):
    cmap = plt.get_cmap('tab10')
    color_index = group_label
    color = cmap(int(color_index))
    return color


n_frame = 541
eps = 2.5
min_samples = 5

with open('../data/people.pkl', 'rb') as f:
    people = pickle.load(f)


images = []
for i in trange(n_frame):
    X = np.array([person.loc for person in people[i]])
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(X)
    labels = model.labels
    fig, ax = plt.subplots(1, figsize=(10, 10))
    for index in range(len(people[i])):
        person = people[i][index]
        color = get_color(labels[index])
        name = "Person" + str(int(person.ID))
        ax.scatter(person.loc[0], person.loc[1],
                   linewidth=2, color=color)
        ax.text(person.loc[0], person.loc[1], name, color=color)
        ax.set_xlim([0, 15])
        ax.set_ylim([0, 15])
        ax.set_title("DBSCAN")

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    image = PIL.Image.open(img_buf)
    images.append(image)
    plt.close()

images[0].save(
    "../video/DBSCAN.gif",
    format="GIF",
    append_images=images[1:],
    save_all=True,
    duration=10,
    loop=0,
)
