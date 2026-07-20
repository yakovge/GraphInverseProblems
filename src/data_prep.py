"""Download and prepare the datasets used by the GRIP paper.

The original `customMETRLA.py` pulls a pre-built ``METR-LA.zip`` from
graphmining.ai, which no longer serves a valid TLS certificate. This module
rebuilds the exact same two artefacts (``adj_mat.npy`` and ``node_values.npy``)
from the canonical DCRNN release, mirrored by torch-spatiotemporal.

The reconstruction follows the standard DCRNN preprocessing so that the
resulting arrays are byte-compatible with what `METRLADatasetLoader` expects:

* ``adj_mat.npy``     -- [207, 207] thresholded Gaussian-kernel adjacency
* ``node_values.npy`` -- [34272, 207, 2], channel 0 = speed, 1 = time-of-day

Run directly to fetch everything:

    python data_prep.py --datapath ../data
"""

import argparse
import os
import zipfile

import numpy as np

# Canonical DCRNN METR-LA release, mirrored by torch-spatiotemporal (tsl).
# The original PyGT host (graphmining.ai) has an expired/broken certificate.
METRLA_URL = "https://drive.switch.ch/index.php/s/Z8cKHAVyiDqkzaG/download"
CPOX_URL = (
    "https://raw.githubusercontent.com/benedekrozemberczki/"
    "pytorch_geometric_temporal/master/dataset/chickenpox.json"
)

# DCRNN sparsifies the Gaussian kernel by zeroing entries below this value.
NORMALIZED_K = 0.1


def _download(url, save_path):
    import urllib.request

    if os.path.isfile(save_path):
        print(f"  already present: {save_path}")
        return save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"  downloading {url}")
    # Some mirrors reject the default urllib user-agent.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(save_path, "wb") as out:
        out.write(resp.read())
    print(f"  saved {save_path} ({os.path.getsize(save_path) / 1e6:.1f} MB)")
    return save_path


def build_metrla_adjacency(distances_csv, sensor_ids_txt):
    """Gaussian-kernel adjacency, thresholded -- the DCRNN construction.

    A_ij = exp(-(d_ij / sigma)^2), zeroed where below NORMALIZED_K, with sigma
    the standard deviation of the observed pairwise road-network distances.
    """
    import pandas as pd

    with open(sensor_ids_txt) as fh:
        sensor_ids = fh.read().strip().split(",")
    id_to_ind = {sid: i for i, sid in enumerate(sensor_ids)}
    n = len(sensor_ids)

    dist_mx = np.full((n, n), np.inf, dtype=np.float32)
    df = pd.read_csv(distances_csv, dtype={"from": "str", "to": "str"})
    for row in df.values:
        src, dst, cost = row[0], row[1], float(row[2])
        if src in id_to_ind and dst in id_to_ind:
            dist_mx[id_to_ind[src], id_to_ind[dst]] = cost

    # sigma over the *observed* (finite) distances only.
    finite = dist_mx[~np.isinf(dist_mx)].flatten()
    sigma = finite.std()

    adj = np.exp(-np.square(dist_mx / sigma))
    adj[adj < NORMALIZED_K] = 0.0
    adj[np.isinf(dist_mx)] = 0.0
    return adj.astype(np.float32)


def _read_metrla_h5(h5_path):
    """The speed matrix [T, N] and its datetime index, without requiring pytables.

    ``pandas.read_hdf`` insists on pytables, but upstream's environment.yml ships h5py
    and NOT pytables -- and building pytables from source needs the HDF5 C headers,
    which is where most environments give up. The fixed-format layout DCRNN wrote
    (one float block under a single group) is simple enough to read with h5py directly.
    """
    try:
        import pandas as pd

        df = pd.read_hdf(h5_path)
        return df.values.astype(np.float32), df.index.values
    except ImportError:
        pass

    try:
        import h5py
    except ImportError:
        raise ImportError(
            "Reading metr_la.h5 needs either pytables (pip install tables) or "
            "h5py (pip install h5py); neither is importable."
        )
    with h5py.File(h5_path, "r") as fh:
        group = fh[next(iter(fh.keys()))]  # DCRNN wrote the frame under 'data'
        values = group["block0_values"][:].astype(np.float32)
        axis0 = [c.decode() for c in group["axis0"][:]]
        items = [c.decode() for c in group["block0_items"][:]]
        if items != axis0:  # align the block's columns to the frame's column order
            order = [items.index(c) for c in axis0]
            values = values[:, order]
        index = group["axis1"][:].astype("datetime64[ns]")
    return values, index


def build_metrla_node_values(h5_path):
    """[T, N, 2] tensor: channel 0 = speed, channel 1 = time-of-day in [0, 1)."""
    speed, idx = _read_metrla_h5(h5_path)  # [T, N]

    # Fraction of the day elapsed, as DCRNN defines it.
    time_in_day = (idx - idx.astype("datetime64[D]")) / np.timedelta64(1, "D")
    time_in_day = np.tile(time_in_day, (speed.shape[1], 1)).T.astype(np.float32)

    return np.stack([speed, time_in_day], axis=-1)  # [T, N, 2]


def prepare_metrla(datapath):
    """Materialise adj_mat.npy / node_values.npy under <datapath>/temporal_data/METRLA."""
    out_dir = os.path.join(datapath, "temporal_data", "METRLA")
    os.makedirs(out_dir, exist_ok=True)

    adj_out = os.path.join(out_dir, "adj_mat.npy")
    val_out = os.path.join(out_dir, "node_values.npy")
    if os.path.isfile(adj_out) and os.path.isfile(val_out):
        print("METR-LA: already prepared")
        return out_dir

    print("METR-LA:")
    zip_path = _download(METRLA_URL, os.path.join(out_dir, "metr_la_source.zip"))

    needed = ["metr_la.h5", "distances_la.csv", "sensor_ids_la.txt"]
    if not all(os.path.isfile(os.path.join(out_dir, f)) for f in needed):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)

    adj = build_metrla_adjacency(
        os.path.join(out_dir, "distances_la.csv"),
        os.path.join(out_dir, "sensor_ids_la.txt"),
    )
    values = build_metrla_node_values(os.path.join(out_dir, "metr_la.h5"))

    np.save(adj_out, adj)
    np.save(val_out, values)

    density = (adj > 0).sum() / adj.size
    print(f"  adj_mat.npy      {adj.shape}  ({density:.1%} dense, {(adj > 0).sum()} edges)")
    print(f"  node_values.npy  {values.shape}")
    return out_dir


def prepare_cpox(datapath):
    """Cache the Chickenpox-Hungary JSON so training runs do not re-fetch it."""
    out_dir = os.path.join(datapath, "temporal_data", "CPOX")
    os.makedirs(out_dir, exist_ok=True)
    print("CPOX:")
    path = _download(CPOX_URL, os.path.join(out_dir, "chickenpox.json"))

    import json

    with open(path) as fh:
        blob = json.load(fh)
    print(f"  nodes={len(blob['FX'][0])}  weeks={len(blob['FX'])}  edges={len(blob['edges'])}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datapath",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"),
        help="root directory to store datasets in",
    )
    args = parser.parse_args()
    datapath = os.path.abspath(args.datapath)
    print(f"datapath = {datapath}\n")

    prepare_metrla(datapath)
    print()
    prepare_cpox(datapath)
    print("\nAll datasets ready.")


if __name__ == "__main__":
    main()
