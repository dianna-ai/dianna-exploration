import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import cv2

with open('test_colours_couples.pk', 'rb') as f:
    couples = pk.load(f)

"""
for couple in couples[:50]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Display the first image on the left subplot
    axes[0].imshow(couple[0], cmap='gray', vmin=0, vmax=255)
    axes[0].axis('off')  # Optional: Turn off axis ticks and labels

    # Display the second image on the right subplot
    axes[1].imshow(couple[1], cmap='inferno')
    axes[1].axis('off')  # Optional: Turn off axis ticks and labels

    # Show the plot
    plt.show()
"""

print(len(couples))

def make_gif():

    for couple in couples:
        colormap = plt.get_cmap('inferno')
        couple[1] = colormap(couple[1])

    frames = [PIL.Image.fromarray(couple[0]) for couple in couples[:100]]
    frame_one = frames[0]
    frame_one.save("my_awesome.gif", format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)

    frames1 = [PIL.Image.fromarray(couple[1]) for couple in couples[:100]]
    frame_one = frames1[0]
    print(len(frames1))
    frame_one.save('my_awesome_1.gif', format="GIF", append_images=frames1,
                   save_all=True, duration=100, loop=0)


if __name__ == "__main__":
    make_gif()