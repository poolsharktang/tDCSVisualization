from simnibs import sim_struct, run_simnibs


# ============================================================
# SimNIBS realistic-head tDCS / tACS demonstration
#
# Uses the MRI-derived "Ernie" example head model.
#
# Current:
#     Anode  = +1 mA
#     Cathode = -1 mA
#
# Outputs:
#     v = voltage
#     e = electric-field magnitude
#     E = electric-field vector
#     j = current-density magnitude
#     J = current-density vector
# ============================================================


# ------------------------------------------------------------
# 1. Create simulation session
# ------------------------------------------------------------

S = sim_struct.SESSION()

# Folder containing the realistic head model
S.subpath = "m2m_ernie"

# Folder where results will be saved
S.pathfem = "tdcs_demo_results"


# ------------------------------------------------------------
# 2. Choose quantities to calculate
# ------------------------------------------------------------

# v = electric potential [V]
# e = |E| electric field magnitude [V/m]
# E = electric field vector [V/m]
# j = |J| current density magnitude [A/m^2]
# J = current density vector [A/m^2]

S.fields = "veEjJ"

# Automatically open the result in Gmsh
S.open_in_gmsh = True


# ------------------------------------------------------------
# 3. Create tDCS simulation
# ------------------------------------------------------------

tdcs = S.add_tdcslist()


# ------------------------------------------------------------
# 4. Define stimulation currents
# ------------------------------------------------------------

# Channel 1: +1 mA
# Channel 2: -1 mA
#
# Total must always sum to zero.

tdcs.currents = [
    +1e-3,
    -1e-3
]


# ------------------------------------------------------------
# 5. Create ANODE
# ------------------------------------------------------------

anode = tdcs.add_electrode()

# Connect to current channel 1
anode.channelnr = 1

# EEG 10-10 position
anode.centre = "C3"

# Circular electrode
# SimNIBS makes a circle by using an ellipse
# with equal X/Y dimensions.
anode.shape = "ellipse"

# 30 mm diameter electrode
anode.dimensions = [30, 30]

# Electrode/gel thickness in mm
anode.thickness = 4


# ------------------------------------------------------------
# 6. Create CATHODE
# ------------------------------------------------------------

cathode = tdcs.add_electrode()

# Connect to current channel 2
cathode.channelnr = 2

# Place second electrode
cathode.centre = "AF4"

cathode.shape = "ellipse"

# 30 mm diameter
cathode.dimensions = [30, 30]

# 4 mm thickness
cathode.thickness = 4


# ------------------------------------------------------------
# 7. Run FEM simulation
# ------------------------------------------------------------

print("Starting SimNIBS simulation...")
print("Anode:   C3, +1 mA")
print("Cathode: AF4, -1 mA")

run_simnibs(S)

print("Simulation finished.")
print("Results saved to:")
print(S.pathfem)