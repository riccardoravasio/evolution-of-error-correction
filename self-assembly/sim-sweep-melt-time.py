#!/usr/bin/env python
# coding: utf-8

# In[20]:


import rgrow as rg
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import math


# # System definitions

# In[2]:


systems = {}


# ## Stalling system with an error pathway (stall-error)

# In[3]:


# %%
# Here, we define the tiles.  We have two repeating rows, each of 5 tiles.  The tile
# definition starts with a list of glues in N, E, S, W order; here, we just set them
# to be placeholders with no matching glues on other tiles; we'll program interactions
# between them next.
tiles = [
    rg.Tile([f"n{i}", f"e{i}", f"s{i}", f"w{i}"], name=f"t{i}", color="gray")
    for i in range(1, 7)
] + [
    rg.Tile([f"n{i}", f"e{i}", f"s{i}", f"w{i}"], name=f"t{i}", color="blue")
    for i in range(7, 13)
]

# Here are our 'standard' glue interactions, which make a ribbon that doesn't stall.
# These are in format (glue1, glue2, strength), where a strength of 1 is normal.
std_gl = (
    [(f"e{i}", f"w{i+6}", 1) for i in range(1, 7)]
    + [(f"e{i}", f"w{i-6}", 1) for i in range(7, 13)]
    + [(f"s{i}", f"n{i+1}", 1) for i in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]]
)

# Here are our 'stalling' glue interactions, which make an out-of-order row.
extra_gl = [
    (f"e{i}", f"w{j}", 1.0) for (i, j) in zip([7, 8, 9, 10, 11, 12], [1, 5, 4, 3, 2, 6], strict=True)
] + [
    (f"s{i}", f"n{j}", 1.0) for (i, j) in zip([1, 5, 4, 3, 2], [5, 4, 3, 2, 6])
]  # removed 7 to prevent periodic annoyance

# Error pathway glues
error_gl = [
    (f"e{i}", f"w{j}", 1) for (i, j) in [(5,8),(3,10)]
]

# Here is our seed, in (x/vertical, y/horizontal, tile_name) format.
# For implementation reasons, no tiles can be placed closer than 2 spaces
# from the boundary of the simulation area.
seed = [(2, 2, "t1"), (3, 2, "t2"), (4, 2, "t3"), (5, 2, "t4"), (6, 2, "t5"), (7, 2, "t6")]

# %%
# Here are our initial options for the simulation.  We'll modify these later.
opts = {
    "size": (10, 128),
    "gse": 9.5,
    "gmc": 16.0,
    "canvas_type": "square",
    "seed": seed,
}

# ts_nostall = rg.TileSet(tiles, glues=std_gl, options=opts)

# "bonds" here refers to strengths of "matching" bonds.  We set all these to 0.
# I should add an option to rgrow to let this be done automatically.
ts = rg.TileSet(
    tiles,
    bonds=[(f"{d}{i}", 0) for d in "nesw" for i in range(1, 11)],
    glues=std_gl + extra_gl+error_gl,
    **opts,
)

systems["stall_error"] = ts


# ## Stalling system with no error pathway (stall-noerror)

# In[4]:


# %%
# Here, we define the tiles.  We have two repeating rows, each of 5 tiles.  The tile
# definition starts with a list of glues in N, E, S, W order; here, we just set them
# to be placeholders with no matching glues on other tiles; we'll program interactions
# between them next.
tiles = [
    rg.Tile([f"n{i}", f"e{i}", f"s{i}", f"w{i}"], name=f"t{i}", color="gray")
    for i in range(1, 7)
] + [
    rg.Tile([f"n{i}", f"e{i}", f"s{i}", f"w{i}"], name=f"t{i}", color="blue")
    for i in range(7, 13)
]

# Here are our 'standard' glue interactions, which make a ribbon that doesn't stall.
# These are in format (glue1, glue2, strength), where a strength of 1 is normal.
std_gl = (
    [(f"e{i}", f"w{i+6}", 1) for i in range(1, 7)]
    + [(f"e{i}", f"w{i-6}", 1) for i in range(7, 13)]
    + [(f"s{i}", f"n{i+1}", 1) for i in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]]
)

# Here are our 'stalling' glue interactions, which make an out-of-order row.
extra_gl = [
    (f"e{i}", f"w{j}", 1.0) for (i, j) in zip([7, 8, 9, 10, 11, 12], [1, 5, 4, 3, 2, 6], strict=True)
] + [
    (f"s{i}", f"n{j}", 1.0) for (i, j) in zip([1, 5, 4, 3, 2], [5, 4, 3, 2, 6])
]  # removed 7 to prevent periodic annoyance

# Error pathway glues
#error_gl = [
#    (f"e{i}", f"w{j}", 1) for (i, j) in [(5,8),(3,10)]
#]

# Here is our seed, in (x/vertical, y/horizontal, tile_name) format.
# For implementation reasons, no tiles can be placed closer than 2 spaces
# from the boundary of the simulation area.
seed = [(2, 2, "t1"), (3, 2, "t2"), (4, 2, "t3"), (5, 2, "t4"), (6, 2, "t5"), (7, 2, "t6")]

# %%
# Here are our initial options for the simulation.  We'll modify these later.
opts = {
    "size": (10, 128),
    "gse": 9.5,
    "gmc": 16.0,
    "canvas_type": "square",
    "seed": seed,
}

# ts_nostall = rg.TileSet(tiles, glues=std_gl, options=opts)

# "bonds" here refers to strengths of "matching" bonds.  We set all these to 0.
# I should add an option to rgrow to let this be done automatically.
ts = rg.TileSet(
    tiles,
    bonds=[(f"{d}{i}", 0) for d in "nesw" for i in range(1, 11)],
    glues=std_gl + extra_gl, # +error_gl,
    **opts,
)

systems["stall_noerror"] = ts


# ## System with an easy error pathway, which should not stall much (littlestall-error)

# In[5]:


# %%
# Here, we define the tiles.  We have two repeating rows, each of 5 tiles.  The tile
# definition starts with a list of glues in N, E, S, W order; here, we just set them
# to be placeholders with no matching glues on other tiles; we'll program interactions
# between them next.
tiles = [
    rg.Tile([f"n{i}", f"e{i}", f"s{i}", f"w{i}"], name=f"t{i}", color="gray")
    for i in range(1, 7)
] + [
    rg.Tile([f"n{i}", f"e{i}", f"s{i}", f"w{i}"], name=f"t{i}", color="blue")
    for i in range(7, 13)
]

# Here are our 'standard' glue interactions, which make a ribbon that doesn't stall.
# These are in format (glue1, glue2, strength), where a strength of 1 is normal.
std_gl = (
    [(f"e{i}", f"w{i+6}", 1) for i in range(1, 7)]
    + [(f"e{i}", f"w{i-6}", 1) for i in range(7, 13)]
    + [(f"s{i}", f"n{i+1}", 1) for i in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]]
)

# Here are our 'stalling' glue interactions, which make an out-of-order row.
extra_gl = [
    (f"e{i}", f"w{j}", 1.0) for (i, j) in zip([7, 8, 9, 10, 11, 12], [1, 5, 4, 3, 2, 6], strict=True)
] + [
    (f"s{i}", f"n{j}", 1.0) for (i, j) in zip([1, 5, 4, 3, 2], [5, 4, 3, 2, 6])
]  # removed 7 to prevent periodic annoyance

# Error pathway glues
error_gl = [
    (f"e{i}", f"w{j}", 1) for (i, j) in [(5,8),(4,9),(3,10)] # ,(2,11)]
]

# Here is our seed, in (x/vertical, y/horizontal, tile_name) format.
# For implementation reasons, no tiles can be placed closer than 2 spaces
# from the boundary of the simulation area.
seed = [(2, 2, "t1"), (3, 2, "t2"), (4, 2, "t3"), (5, 2, "t4"), (6, 2, "t5"), (7, 2, "t6")]

# %%
# Here are our initial options for the simulation.  We'll modify these later.
opts = {
    "size": (10, 128),
    "gse": 9.5,
    "gmc": 16.0,
    "canvas_type": "square",
    "seed": seed,
}

# ts_nostall = rg.TileSet(tiles, glues=std_gl, options=opts)

# "bonds" here refers to strengths of "matching" bonds.  We set all these to 0.
# I should add an option to rgrow to let this be done automatically.
ts = rg.TileSet(
    tiles,
    bonds=[(f"{d}{i}", 0) for d in "nesw" for i in range(1, 11)],
    glues=std_gl + extra_gl +error_gl,
    **opts,
)

#systems["littlestall_error"] = ts


# ## System with no stalling at all in the error pathway (nostall-error)

# In[6]:


# %%
# Here, we define the tiles.  We have two repeating rows, each of 5 tiles.  The tile
# definition starts with a list of glues in N, E, S, W order; here, we just set them
# to be placeholders with no matching glues on other tiles; we'll program interactions
# between them next.
tiles = [
    rg.Tile([f"n{i}", f"e{i}", f"s{i}", f"w{i}"], name=f"t{i}", color="gray")
    for i in range(1, 7)
] + [
    rg.Tile([f"n{i}", f"e{i}", f"s{i}", f"w{i}"], name=f"t{i}", color="blue")
    for i in range(7, 13)
]

# Here are our 'standard' glue interactions, which make a ribbon that doesn't stall.
# These are in format (glue1, glue2, strength), where a strength of 1 is normal.
std_gl = (
    [(f"e{i}", f"w{i+6}", 1) for i in range(1, 7)]
    + [(f"e{i}", f"w{i-6}", 1) for i in range(7, 13)]
    + [(f"s{i}", f"n{i+1}", 1) for i in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]]
)

# Here are our 'stalling' glue interactions, which make an out-of-order row.
extra_gl = [
    (f"e{i}", f"w{j}", 1.0) for (i, j) in zip([7, 8, 9, 10, 11, 12], [1, 5, 4, 3, 2, 6], strict=True)
] + [
    (f"s{i}", f"n{j}", 1.0) for (i, j) in zip([1, 5, 4, 3, 2], [5, 4, 3, 2, 6])
]  # removed 7 to prevent periodic annoyance

# Error pathway glues
error_gl = [
    (f"e{i}", f"w{j}", 1) for (i, j) in [(5,8),(4,9),(3,10),(2,11)]
]

# Here is our seed, in (x/vertical, y/horizontal, tile_name) format.
# For implementation reasons, no tiles can be placed closer than 2 spaces
# from the boundary of the simulation area.
seed = [(2, 2, "t1"), (3, 2, "t2"), (4, 2, "t3"), (5, 2, "t4"), (6, 2, "t5"), (7, 2, "t6")]

# %%
# Here are our initial options for the simulation.  We'll modify these later.
opts = {
    "size": (10, 128),
    "gse": 9.5,
    "gmc": 16.0,
    "canvas_type": "square",
    "seed": seed,
}

# ts_nostall = rg.TileSet(tiles, glues=std_gl, options=opts)

# "bonds" here refers to strengths of "matching" bonds.  We set all these to 0.
# I should add an option to rgrow to let this be done automatically.
ts = rg.TileSet(
    tiles,
    bonds=[(f"{d}{i}", 0) for d in "nesw" for i in range(1, 11)],
    glues=std_gl + extra_gl +error_gl,
    **opts,
)

systems["nostall_error"] = ts


# ## System with medium stalling in the error pathway (mediumstall-error)

# In[7]:


# %%
# Here, we define the tiles.  We have two repeating rows, each of 5 tiles.  The tile
# definition starts with a list of glues in N, E, S, W order; here, we just set them
# to be placeholders with no matching glues on other tiles; we'll program interactions
# between them next.
tiles = [
    rg.Tile([f"n{i}", f"e{i}", f"s{i}", f"w{i}"], name=f"t{i}", color="gray")
    for i in range(1, 7)
] + [
    rg.Tile([f"n{i}", f"e{i}", f"s{i}", f"w{i}"], name=f"t{i}", color="blue")
    for i in range(7, 13)
]

# Here are our 'standard' glue interactions, which make a ribbon that doesn't stall.
# These are in format (glue1, glue2, strength), where a strength of 1 is normal.
std_gl = (
    [(f"e{i}", f"w{i+6}", 1) for i in range(1, 7)]
    + [(f"e{i}", f"w{i-6}", 1) for i in range(7, 13)]
    + [(f"s{i}", f"n{i+1}", 1) for i in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]]
)

# Here are our 'stalling' glue interactions, which make an out-of-order row.
extra_gl = [
    (f"e{i}", f"w{j}", 1.0) for (i, j) in zip([7, 8, 9, 10, 11, 12], [1, 5, 4, 3, 2, 6], strict=True)
] + [
    (f"s{i}", f"n{j}", 1.0) for (i, j) in zip([1, 5, 4, 3, 2], [5, 4, 3, 2, 6])
]  # removed 7 to prevent periodic annoyance

# Error pathway glues
error_gl = [
    (f"e{i}", f"w{j}", 1) for (i, j) in [(5,8),(4,9)]
]

# Here is our seed, in (x/vertical, y/horizontal, tile_name) format.
# For implementation reasons, no tiles can be placed closer than 2 spaces
# from the boundary of the simulation area.
seed = [(2, 2, "t1"), (3, 2, "t2"), (4, 2, "t3"), (5, 2, "t4"), (6, 2, "t5"), (7, 2, "t6")]

# %%
# Here are our initial options for the simulation.  We'll modify these later.
opts = {
    "size": (10, 128),
    "gse": 9.5,
    "gmc": 16.0,
    "canvas_type": "square",
    "seed": seed,
}

# ts_nostall = rg.TileSet(tiles, glues=std_gl, options=opts)

# "bonds" here refers to strengths of "matching" bonds.  We set all these to 0.
# I should add an option to rgrow to let this be done automatically.
ts = rg.TileSet(
    tiles,
    bonds=[(f"{d}{i}", 0) for d in "nesw" for i in range(1, 11)],
    glues=std_gl + extra_gl +error_gl,
    **opts,
)

systems["mediumstall_error"] = ts


# In[8]:


# %%
# Here, we define the tiles.  We have two repeating rows, each of 5 tiles.  The tile
# definition starts with a list of glues in N, E, S, W order; here, we just set them
# to be placeholders with no matching glues on other tiles; we'll program interactions
# between them next.
tiles = [
    rg.Tile([f"n{i}", f"e{i}", f"s{i}", f"w{i}"], name=f"t{i}", color="gray")
    for i in range(1, 7)
] + [
    rg.Tile([f"n{i}", f"e{i}", f"s{i}", f"w{i}"], name=f"t{i}", color="blue")
    for i in range(7, 13)
]

# Here are our 'standard' glue interactions, which make a ribbon that doesn't stall.
# These are in format (glue1, glue2, strength), where a strength of 1 is normal.
std_gl = (
    [(f"e{i}", f"w{i+6}", 1) for i in range(1, 7)]
    + [(f"e{i}", f"w{i-6}", 1) for i in range(7, 13)]
    + [(f"s{i}", f"n{i+1}", 1) for i in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]]
)

# Here are our 'stalling' glue interactions, which make an out-of-order row.
extra_gl = [
    (f"e{i}", f"w{j}", 1.0) for (i, j) in zip([7, 8, 9, 10, 11, 12], [1, 5, 4, 3, 2, 6], strict=True)
] + [
    (f"s{i}", f"n{j}", 1.0) for (i, j) in zip([1, 5, 4, 3, 2], [5, 4, 3, 2, 6])
]  # removed 7 to prevent periodic annoyance

# Error pathway glues
error_gl = [
    (f"e{i}", f"w{j}", 1) for (i, j) in [(5,8),(2,11)]
]

# Here is our seed, in (x/vertical, y/horizontal, tile_name) format.
# For implementation reasons, no tiles can be placed closer than 2 spaces
# from the boundary of the simulation area.
seed = [(2, 2, "t1"), (3, 2, "t2"), (4, 2, "t3"), (5, 2, "t4"), (6, 2, "t5"), (7, 2, "t6")]

# %%
# Here are our initial options for the simulation.  We'll modify these later.
opts = {
    "size": (10, 128),
    "gse": 9.5,
    "gmc": 16.0,
    "canvas_type": "square",
    "seed": seed,
}

# ts_nostall = rg.TileSet(tiles, glues=std_gl, options=opts)

# "bonds" here refers to strengths of "matching" bonds.  We set all these to 0.
# I should add an option to rgrow to let this be done automatically.
ts = rg.TileSet(
    tiles,
    bonds=[(f"{d}{i}", 0) for d in "nesw" for i in range(1, 11)],
    glues=std_gl + extra_gl +error_gl,
    **opts,
)

# systems["mediumplusstall_error"] = ts


# In[9]:


systems


# # Simulation code

# In[10]:


def varsims(
    ts: rg.TileSet,
    gsemelt,
    melttime,
    meltperiod,
    gsegrow=9.5,
    timelimit=1e7,
    smax=200,
    nsims=20,
    maxcycle=None
):
    if maxcycle is None:
        maxcycle = math.ceil(timelimit / meltperiod)*2
    do_sims = list(range(0, nsims))
    system = ts.create_system()
    states = [ts.create_state(system) for _ in range(nsims)]
    times = [[] for _ in range(nsims)]
    sizes = [[] for _ in range(nsims)]
    mismatches = [[] for _ in range(nsims)]
    lastcycle = np.zeros(nsims, dtype=int)
    cycle = 0
    while do_sims:
        u = system.set_param("g_se", gsemelt)
        for state in states:
            system.update_all(state, u)
        results = system.evolve_states(
            [states[i] for i in do_sims], size_max=smax, for_time=melttime
        )
        for i in do_sims:
            times[i].append(states[i].time)
            sizes[i].append(states[i].ntiles)
            mismatches[i].append(system.calc_mismatches(states[i]))
        cycle += 1
        do_sims = [
            i
            for i, r in zip(do_sims, results)
            if (r != rg.EvolveOutcome.ReachedSizeMax) and (states[i].time < timelimit)
        ]
        lastcycle[do_sims] = cycle
        u = system.set_param("g_se", gsegrow)
        for state in states:
            system.update_all(state, u)
        results = system.evolve_states(
            [states[i] for i in do_sims], size_max=smax, for_time=meltperiod - melttime
        )
        for i in do_sims:
            times[i].append(states[i].time)
            sizes[i].append(states[i].ntiles)
            mismatches[i].append(system.calc_mismatches(states[i]))
        cycle += 1
        do_sims = [
            i
            for i, r in zip(do_sims, results)
            if (r != rg.EvolveOutcome.ReachedSizeMax) and (states[i].time < timelimit)
        ]
        lastcycle[do_sims] = cycle
    invtimes = np.array([1 / state.time for state in states])
    mms = np.array([system.calc_mismatches(state) for state in states])
    return {
        "invtime": np.mean(invtimes),
        "mm": np.mean(mms),
        "invtimes_array": invtimes,
        "mms_array": mms,
        "times_array": times,
        "sizes_array": sizes,
        "mismatches_array": mismatches,
        "lastcycle": lastcycle,
        "system": system,
        "states": states,
    }


# In[11]:


import tables as tb

def createsimres(nsims=20, canvassize=(10, 128)):
    class SimResults(tb.IsDescription):
        # mutindex = tb.Int64Col()
        growgrse = tb.Float64Col()
        gsemelt = tb.Float64Col()
        melttime = tb.Float64Col()
        meltperiod = tb.Float64Col()
        invtime_avg = tb.Float64Col()
        mm_avg = tb.Float64Col()
        
        invtimes_array = tb.Float64Col(shape=(nsims))
        mms_array = tb.Float64Col(shape=(nsims))
        lastcycle = tb.Int64Col(shape=(nsims))
        
        final_tiles_array = tb.UInt8Col(shape=(nsims, canvassize[0], canvassize[1]))
        mismatch_locs_array = tb.UInt8Col(shape=(nsims, canvassize[0], canvassize[1]))
        
    return SimResults

class SimTrace(tb.IsDescription):
    times = tb.Float64Col()
    sizes = tb.UInt64Col()
    mismatches = tb.UInt64Col()

# In[12]:


import datetime


# In[13]:


def sweep_melttime(
    ts: rg.TileSet,
    melttime: np.ndarray,
    meltgse: float,
    output_file: tb.File,
    output_group: str = "simresults",
    output_name: str = "simresults",
    trace_group: str = "traces",
    evolve_melttime: float | None = 15.0,
    maxtime: float = 1e7,
    nsims: int = 20,
    nsteps: int = 500,
    growgse: float = 9.5,
    meltperiod: float = 3600,
    targetsize: int = 200,
    cutcol: int | None = None,
):
    desc = createsimres(nsims=nsims, canvassize=ts.size)
    restable = output_file.create_table(
        output_group,
        output_name,
        desc,
        expectedrows=nsteps,
    )
    row = restable.row
    
    init_meltgse = meltgse
    init_melttime = melttime
    
    bestiv = 0
    bestmm = 0
    besti = 0
    
    restable.attrs["growgse"] = growgse
    restable.attrs["meltperiod"] = meltperiod
    restable.attrs["targetsize"] = targetsize
    restable.attrs["nsims"] = nsims
    restable.attrs["maxtime"] = maxtime
    restable.attrs["nsteps"] = nsteps
    restable.attrs["melttime_values"] = melttime
    restable.attrs["meltgse"] = meltgse
    restable.attrs["cutcol"] = cutcol
    restable.attrs["run_start"] = datetime.datetime.now().isoformat()
    #restable.attrs["ts"] = ts.to_json()

    out_group = output_file.get_node(output_group)
    trace_group_node = output_file.create_group(out_group, trace_group)
    
    
    for melttime_val in melttime:
        res = varsims(
            ts,
            gsemelt,
            melttime_val,
            meltperiod, # trial_period,
            timelimit=maxtime,
            smax=targetsize, # 400,
            nsims=nsims, # 200 # up from 100 for rna-2 and lazarus-2
        )
        row["gsemelt"] = gsemelt
        row["melttime"] = melttime_val
        row["meltperiod"] = meltperiod
        row["invtime_avg"] = res["invtime"]
        row["mm_avg"] = res["mm"]
        row["invtimes_array"] = res["invtimes_array"]
        row["mms_array"] = res["mms_array"]
        
        row["lastcycle"] = res["lastcycle"]
        
        for j in range(nsims):
            trace_table = h5f.create_table(
                trace_group_node,
                f"trace_{melttime_val:.0f}_{j}",
                SimTrace,
                expectedrows=res["lastcycle"][j],
            )
            trace_table.append([(res["times_array"][j][k], res["sizes_array"][j][k], res["mismatches_array"][j][k]) for k in range(res["lastcycle"][j])])
            trace_table.flush()
        
        row["final_tiles_array"] = np.array([state.canvas_view for state in res["states"]])
        row["mismatch_locs_array"] = np.array([res["system"].calc_mismatch_locations(state) for state in res['states']])
        row.append()
        restable.flush()

        print(
            f"{gsemelt:.3f}\t{meltperiod:.2f}\t{melttime_val:.2f}\t{res['invtime']:.3e}\t{res['mm']:.3f}",
            flush=True,
        )
    return h5f
        


# In[25]:


with tb.open_file("2023-11-21_sims-sweep-melt-time-big.h5", "w", filters=tb.Filters(complevel=5, complib='blosc2:zstd')) as h5f:
    for s in ["stall_error", "mediumstall_error", "stall_noerror"]:
        group = h5f.create_group(f"/", f"{s}", f"System {s}", createparents=True)
        group.system = s
    for gsemelt in [8.4, 8.3]:
        for s in ["stall_error", "mediumstall_error", "stall_noerror"]:
            print(f"Starting {s} melttime sweep at gsemelt={gsemelt}:")
            sweep_melttime(systems[s], np.arange(0,500,5), gsemelt, output_file=h5f,
                    output_group=f"/{s}", output_name=f"results_{gsemelt}",
                    trace_group=f"traces_{gsemelt}", nsims=12*4*4*4,
                    targetsize=12*40, maxtime=1e8)


# # Scratch




