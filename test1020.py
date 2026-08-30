import mne
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------
# Load standard EEG montage
# --------------------------------------------------

montage = mne.channels.make_standard_montage("standard_1020")

positions = montage.get_positions()["ch_pos"]


# --------------------------------------------------
# Choose the electrodes we want to display
#
# You can expand this list later.
# --------------------------------------------------

channels = [
    "Fp1", "Fpz", "Fp2",

    "F7", "F3", "Fz", "F4", "F8",

    "FC5", "FC3", "FC1",
    "FCz",
    "FC2", "FC4", "FC6",

    "T7", "C5", "C3", "C1",
    "Cz",
    "C2", "C4", "C6", "T8",

    "CP5", "CP3", "CP1",
    "CPz",
    "CP2", "CP4", "CP6",

    "P7", "P3", "P1",
    "Pz",
    "P2", "P4", "P8",

    "PO7", "PO3", "POz", "PO4", "PO8",

    "O1", "Oz", "O2",
]


# --------------------------------------------------
# Extract XY coordinates
# --------------------------------------------------

xy = {}

for ch in channels:

    if ch in positions:

        xyz = positions[ch]

        xy[ch] = np.array([
            xyz[0],
            xyz[1]
        ])


# --------------------------------------------------
# Normalize coordinates for GUI display
# --------------------------------------------------

all_xy = np.array(list(xy.values()))

scale = np.max(np.abs(all_xy))

for ch in xy:
    xy[ch] = xy[ch] / scale


# --------------------------------------------------
# State
# --------------------------------------------------

sources = set()
sinks = set()

selection_mode = "source"


# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 8))


def redraw():

    ax.clear()

    # Head circle
    head = plt.Circle(
        (0, 0),
        1.08,
        fill=False,
        linewidth=2
    )

    ax.add_patch(head)

    # Nose
    ax.plot(
        [-0.10, 0, 0.10],
        [1.08, 1.18, 1.08],
        linewidth=2
    )

    # Electrodes
    for ch, pos in xy.items():

        x, y = pos

        if ch in sources:

            color = "red"

        elif ch in sinks:

            color = "blue"

        else:

            color = "lightgray"

        ax.scatter(
            x,
            y,
            s=350,
            color=color,
            edgecolor="black",
            zorder=3
        )

        ax.text(
            x,
            y,
            ch,
            ha="center",
            va="center",
            fontsize=8,
            zorder=4
        )

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)

    ax.set_aspect("equal")

    ax.axis("off")

    ax.set_title(
        f"Selection mode: {selection_mode.upper()}"
    )

    fig.canvas.draw_idle()


# --------------------------------------------------
# Mouse click
# --------------------------------------------------

def onclick(event):

    if event.xdata is None or event.ydata is None:
        return

    click = np.array([
        event.xdata,
        event.ydata
    ])

    # Find nearest electrode
    nearest = None
    nearest_distance = 999

    for ch, pos in xy.items():

        distance = np.linalg.norm(
            click - pos
        )

        if distance < nearest_distance:

            nearest = ch
            nearest_distance = distance

    # Ignore click if too far away
    if nearest_distance > 0.10:
        return

    if selection_mode == "source":

        sources.add(nearest)
        sinks.discard(nearest)

    elif selection_mode == "sink":

        sinks.add(nearest)
        sources.discard(nearest)

    print()
    print("SOURCE:", sorted(sources))
    print("SINK:", sorted(sinks))

    redraw()


fig.canvas.mpl_connect(
    "button_press_event",
    onclick
)


redraw()

plt.show()