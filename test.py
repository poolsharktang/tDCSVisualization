import matplotlib.pyplot as plt
import numpy as np
import pyeit.mesh as mesh
from pyeit.eit.fem import Forward


# ---------------------------------------------------------
# 1. Create a 2D circular mesh
# ---------------------------------------------------------

N_ELECTRODES = 16
SIGMA = 1.0        # uniform conductivity
H0 = 0.06          # smaller = finer mesh

# mesh.create() already uses a unit circle by default
mesh_obj = mesh.create(
    n_el=N_ELECTRODES,
    h0=H0
)

# Set uniform conductivity
mesh_obj.perm = SIGMA


# ---------------------------------------------------------
# 2. Select stimulation electrodes
# ---------------------------------------------------------

# Electrode 0 = source/anode
# Electrode 8 = sink/cathode
#
# For 16 equally spaced electrodes, these are opposite
# sides of the circle.
electrode_pair = np.array([0, 8], dtype=int)


# ---------------------------------------------------------
# 3. Set up FEM forward solver
# ---------------------------------------------------------

# Current pyEIT Forward() only takes the mesh.
fwd = Forward(mesh_obj)


# ---------------------------------------------------------
# 4. Solve node potentials
# ---------------------------------------------------------

# pyEIT internally applies:
# +1 current at electrode 0
# -1 current at electrode 8
#
# solve() returns voltage/potential at every FEM node.
phi = fwd.solve(electrode_pair)


# ---------------------------------------------------------
# 5. Get mesh geometry
# ---------------------------------------------------------

tri = mesh_obj.element

# pyEIT may store nodes as [x, y, z], even for a 2-D mesh.
# We only want x and y.
pts = mesh_obj.node[:, :2]

n_tri = tri.shape[0]


# ---------------------------------------------------------
# 6. Calculate current density in each triangle
#
# J = -sigma * grad(V)
# ---------------------------------------------------------

jx = np.zeros(n_tri)
jy = np.zeros(n_tri)

xc = np.zeros(n_tri)
yc = np.zeros(n_tri)

# Conductivity for each FEM element
sigma_elem = mesh_obj.perm_array


for i, t in enumerate(tri):

    # Coordinates of triangle vertices
    p = pts[t]

    # Potentials at triangle vertices
    v = phi[t]

    # Triangle center
    xc[i], yc[i] = np.mean(p, axis=0)

    # Inside a linear triangular FEM element:
    #
    # V(x,y) = a*x + b*y + c
    #
    # Therefore:
    #
    # grad(V) = [a, b]

    A = np.column_stack(
        (
            p[:, 0],
            p[:, 1],
            np.ones(3)
        )
    )

    try:
        coef = np.linalg.solve(A, v)

        dVdx = coef[0]
        dVdy = coef[1]

        # Current density:
        # J = -sigma * grad(V)

        jx[i] = -sigma_elem[i] * dVdx
        jy[i] = -sigma_elem[i] * dVdy

    except np.linalg.LinAlgError:
        # Degenerate triangle -- should normally not occur
        jx[i] = 0
        jy[i] = 0


# ---------------------------------------------------------
# 7. Calculate magnitude of current density
# ---------------------------------------------------------

magnitude = np.sqrt(jx**2 + jy**2)

# Avoid divide-by-zero when normalizing arrows
magnitude_safe = magnitude.copy()
magnitude_safe[magnitude_safe == 0] = 1.0

jx_norm = jx / magnitude_safe
jy_norm = jy / magnitude_safe


# ---------------------------------------------------------
# 8. Visualization
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 8))


# Potential distribution
im = ax.tripcolor(
    pts[:, 0],
    pts[:, 1],
    tri,
    phi,
    cmap="coolwarm",
    shading="gouraud"
)

fig.colorbar(
    im,
    ax=ax,
    label="Electric Potential"
)


# ---------------------------------------------------------
# Current flow vectors
# ---------------------------------------------------------

# Plot every few vectors if the mesh is very dense
skip = 3

ax.quiver(
    xc[::skip],
    yc[::skip],
    jx_norm[::skip],
    jy_norm[::skip],
    color="black",
    alpha=0.65,
    scale=30,
    width=0.0025
)


# ---------------------------------------------------------
# 9. Highlight stimulation electrodes
# ---------------------------------------------------------

# electrode_pair contains electrode NUMBERS.
# mesh_obj.el_pos converts those into mesh NODE numbers.
electrode_nodes = mesh_obj.el_pos[electrode_pair]

anode_node = electrode_nodes[0]
cathode_node = electrode_nodes[1]

ax.plot(
    pts[anode_node, 0],
    pts[anode_node, 1],
    "ro",
    markersize=12,
    label="Anode (+)"
)

ax.plot(
    pts[cathode_node, 0],
    pts[cathode_node, 1],
    "bo",
    markersize=12,
    label="Cathode (-)"
)


# ---------------------------------------------------------
# Plot formatting
# ---------------------------------------------------------

ax.set_title(
    "tDCS / tACS FEM Simulation\n"
    "Electric Potential and Current Flow"
)

ax.set_aspect("equal")
ax.legend()

ax.set_xlabel("x")
ax.set_ylabel("y")

plt.tight_layout()
plt.show()