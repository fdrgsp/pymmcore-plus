#!/usr/bin/env python3

"""Simple test script to verify OME-Zarr functionality in TensorStoreHandler."""

import json
import tempfile
from pathlib import Path

import numpy as np
import useq

from pymmcore_plus.mda.handlers import TensorStoreHandler


def test_ome_zarr_functionality():
    """Test TensorStoreHandler with OME-Zarr metadata emission."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        zarr_path = Path(tmp_dir) / "test_ome.zarr"

        # Create handler with OME-Zarr enabled
        handler = TensorStoreHandler(
            path=zarr_path, driver="zarr", ome_zarr=True, array_path="0"
        )

        # Create a simple sequence
        sequence = useq.MDASequence(
            time_plan=useq.TIntervalLoops(interval=0.1, loops=2),
            channels=[useq.Channel(config="FITC")],
            z_plan=useq.ZRangeAround(range=2, step=1),
            stage_positions=[useq.Position(x=0, y=0)],
        )

        # Start sequence
        handler.sequenceStarted(sequence, {})

        # Add some mock frames
        for _, event in enumerate(sequence):
            frame = np.random.randint(0, 255, (256, 256), dtype=np.uint16)
            meta = {
                "Width": 256,
                "Height": 256,
                "pixel_size_x": 0.65,
                "pixel_size_y": 0.65,
            }
            handler.frameReady(frame, event, meta)

        # Finish sequence
        handler.sequenceFinished(sequence)

        # Check if OME-Zarr files were created
        zarr_path = Path(zarr_path)
        print(f"Zarr path exists: {zarr_path.exists()}")

        if zarr_path.exists():
            print("Files in zarr directory:")
            for f in zarr_path.rglob("*"):
                print(f"  {f.relative_to(zarr_path)}")

            # Check for OME-Zarr specific files
            zgroup_file = zarr_path / ".zgroup"
            zattrs_file = zarr_path / ".zattrs"

            print(f"\n.zgroup exists: {zgroup_file.exists()}")
            print(f".zattrs exists: {zattrs_file.exists()}")

            if zattrs_file.exists():
                with open(zattrs_file) as f:
                    attrs = json.load(f)
                print("\n.zattrs content:")
                print(json.dumps(attrs, indent=2))

                # Check for multiscales key
                if "multiscales" in attrs:
                    print("\n✅ OME-Zarr metadata found!")
                    multiscales = attrs["multiscales"][0]
                    print(f"Version: {multiscales.get('version')}")
                    print(f"Axes: {[ax['name'] for ax in multiscales.get('axes', [])]}")
                else:
                    print("\n❌ No multiscales metadata found")
        else:
            print("❌ Zarr directory was not created")


if __name__ == "__main__":
    test_ome_zarr_functionality()
