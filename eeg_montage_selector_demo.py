import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
import numpy as np
import mne

# ============================================================
# EEG 10-10 Montage Selector Demo
#
# Red  = Source
# Blue = Sink
# Gray = Unselected
#
# Install:
#   pip install mne matplotlib numpy
#
# Run:
#   python eeg_montage_selector_demo.py
# ============================================================

# 1. Load standard EEG montage
montage = mne.channels.make_standard_montage("standard_1020")
positions_3d = montage.get_positions()["ch_pos"]

# 2. Electrode set to display
channels = [
    "Fp1", "Fpz", "Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
]

# 3. Convert MNE 3D positions to 2D top view
raw_xy = {}
for ch in channels:
    if ch in positions_3d:
        xyz = positions_3d[ch]
        raw_xy[ch] = np.array([xyz[0], xyz[1]], dtype=float)
    else:
        print(f"Warning: {ch} not found in MNE montage")

names = list(raw_xy.keys())
coords = np.array([raw_xy[ch] for ch in names])

# Center and normalize
center = np.mean(coords, axis=0)
coords = coords - center
radius = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
coords = coords / np.max(radius)

xy = {ch: coords[i] for i, ch in enumerate(names)}

# 4. Selection state
sources = set()
sinks = set()
selection_mode = "source"

# 5. Main figure
fig, ax = plt.subplots(figsize=(11, 8))
plt.subplots_adjust(left=0.04, right=0.74, top=0.93, bottom=0.06)

# 6. Right-side controls
fig.text(0.79, 0.88, "Selection Mode", fontsize=13, fontweight="bold")

radio_ax = plt.axes([0.79, 0.68, 0.17, 0.16])
radio = RadioButtons(radio_ax, ("Source", "Sink", "Remove"), active=0)

clear_ax = plt.axes([0.79, 0.58, 0.17, 0.055])
clear_button = Button(clear_ax, "Clear All")

fig.text(0.79, 0.50, "Current Selection", fontsize=13, fontweight="bold")

source_text = fig.text(
    0.79, 0.45, "Source:\nNone",
    fontsize=10, va="top"
)

sink_text = fig.text(
    0.79, 0.30, "Sink:\nNone",
    fontsize=10, va="top"
)

fig.text(
    0.79, 0.10,
    "Red   = Source\n"
    "Blue  = Sink\n"
    "Gray  = Unselected\n\n"
    "Choose a mode,\n"
    "then click electrodes.",
    fontsize=9,
    va="top",
)

# 7. Redraw function
def redraw():
    ax.clear()

    # Head outline
    head = plt.Circle((0, 0), 1.08, fill=False, linewidth=2.2, color="black")
    ax.add_patch(head)

    # Nose
    ax.plot(
        [-0.10, 0.0, 0.10],
        [1.07, 1.18, 1.07],
        color="black",
        linewidth=2,
    )

    # Ears
    ax.plot(
        [-1.08, -1.15, -1.16, -1.08],
        [0.15, 0.10, -0.10, -0.15],
        color="black",
        linewidth=2,
    )
    ax.plot(
        [1.08, 1.15, 1.16, 1.08],
        [0.15, 0.10, -0.10, -0.15],
        color="black",
        linewidth=2,
    )

    # Electrodes
    for ch, pos in xy.items():
        x, y = pos

        if ch in sources:
            facecolor = "red"
            textcolor = "white"
        elif ch in sinks:
            facecolor = "blue"
            textcolor = "white"
        else:
            facecolor = "lightgray"
            textcolor = "black"

        ax.scatter(
            x, y,
            s=380,
            color=facecolor,
            edgecolor="black",
            linewidth=1.0,
            zorder=3,
        )

        ax.text(
            x, y, ch,
            ha="center",
            va="center",
            fontsize=7.5,
            color=textcolor,
            fontweight="bold",
            zorder=4,
        )

    ax.set_xlim(-1.28, 1.28)
    ax.set_ylim(-1.28, 1.28)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("EEG 10-10 Montage Selector", fontsize=16, fontweight="bold")

    source_string = ", ".join(sorted(sources)) if sources else "None"
    sink_string = ", ".join(sorted(sinks)) if sinks else "None"

    source_text.set_text("Source:\n" + source_string)
    sink_text.set_text("Sink:\n" + sink_string)

    fig.canvas.draw_idle()

# 8. Change mode
def change_mode(label):
    global selection_mode

    if label == "Source":
        selection_mode = "source"
    elif label == "Sink":
        selection_mode = "sink"
    elif label == "Remove":
        selection_mode = "remove"

    print(f"Selection mode: {selection_mode.upper()}")

radio.on_clicked(change_mode)

# 9. Electrode click handler
def onclick(event):
    if event.inaxes != ax:
        return

    if event.xdata is None or event.ydata is None:
        return

    click = np.array([event.xdata, event.ydata])

    nearest = None
    nearest_distance = np.inf

    for ch, pos in xy.items():
        distance = np.linalg.norm(click - pos)

        if distance < nearest_distance:
            nearest = ch
            nearest_distance = distance

    # Ignore clicks that are not close enough to an electrode
    if nearest is None or nearest_distance > 0.085:
        return

    if selection_mode == "source":
        sources.add(nearest)
        sinks.discard(nearest)

    elif selection_mode == "sink":
        sinks.add(nearest)
        sources.discard(nearest)

    elif selection_mode == "remove":
        sources.discard(nearest)
        sinks.discard(nearest)

    print()
    print("===================================")
    print("Current Montage")
    print("===================================")
    print("SOURCE:", ", ".join(sorted(sources)) if sources else "None")
    print("SINK:", ", ".join(sorted(sinks)) if sinks else "None")

    redraw()

fig.canvas.mpl_connect("button_press_event", onclick)

# 10. Clear all
def clear_all(event):
    sources.clear()
    sinks.clear()
    print("All electrode selections cleared.")
    redraw()

clear_button.on_clicked(clear_all)

# 11. Initial draw
redraw()
plt.show()
