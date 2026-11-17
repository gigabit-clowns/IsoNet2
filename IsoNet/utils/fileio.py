
import mrcfile
import numpy as np
import logging
import shutil
import os
import shutil

def create_folder(folder, remove=True):
    try:
        os.makedirs(folder)
    except FileExistsError:
        if remove:
            logging.warning(f"The {folder} folder already exists. The old {folder} folder will be moved to {folder}~")
            if os.path.exists(folder+'~'):
                shutil.rmtree(folder+'~')
            shutil.move(folder, folder+'~')
            os.makedirs(folder)
        else:
            logging.info(f"The {folder} folder already exists, outputs will write into this folder")

def read_mrc(filename, inplace=False):
    if inplace:
        with mrcfile.mmap(filename, permissive=True) as mrc:
            data = mrc.data
            voxel_size = mrc.voxel_size.x
    else:
        with mrcfile.open(filename, permissive=True) as mrc:
            data = mrc.data
            voxel_size = mrc.voxel_size.x
    return data, voxel_size

def write_mrc(filename, data, voxel_size=1, origin=0):
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))
        mrc.voxel_size = voxel_size
        mrc.header.origin = origin

def write_mrc_sections(filename, shape, sections, voxel_size=1, origin=0):
    """Write a volume of `shape` as write_mrc does, a stretch of sections along z at a
    time as `sections` yields them, each as the index of its first section and its
    values, while the next ones are made.

    The values go through vitrio, the header as mrcfile writes it, with the statistics
    of the values gathered as they are written."""
    import vitrio as vio

    writers = vio.writer_provider()
    writers.declare(filename, vio.ImageDescriptor(tuple(shape), 3, vio.NumericalType.float32))
    saver = vio.saver(writers, workers=2)
    location = vio.ImageLocation(filename)
    statistics = _Statistics()
    pending = None
    for first, values in sections:
        values = np.ascontiguousarray(values, dtype=np.float32)
        statistics.add(values)
        # a patch spanning the sections whole, centred where a read of them is
        centre = (first + len(values) // 2, shape[1] // 2, shape[2] // 2)
        writing = vio.write_patches_async(saver, values[np.newaxis], location, [centre])
        if pending is not None:
            pending.get()
        pending = writing
    if pending is not None:
        pending.get()
    writers.close(filename)
    if statistics.count != np.prod(shape):
        raise RuntimeError(f"{statistics.count} of the {np.prod(shape)} voxels of {filename} were written.")

    with mrcfile.mmap(filename, mode='r+', permissive=True) as mrc:
        mrc.voxel_size = voxel_size
        mrc.header.origin = origin
        mrc.header.dmin, mrc.header.dmax, mrc.header.dmean, mrc.header.rms = statistics.header()


class _Statistics:
    """The minimum, maximum, mean and standard deviation of values seen a part at a
    time, as mrcfile puts them in a header, the moments combined as Chan et al. do."""

    def __init__(self):
        self.count, self.mean, self.squares = 0, 0.0, 0.0
        self.minimum, self.maximum = np.inf, -np.inf

    def add(self, values):
        n = values.size
        mean = values.mean(dtype=np.float64)
        squares = np.square(values - mean, dtype=np.float64).sum()
        delta, total = mean - self.mean, self.count + n
        self.squares += squares + delta * delta * self.count * n / total
        self.mean += delta * n / total
        self.count = total
        self.minimum = min(self.minimum, values.min())
        self.maximum = max(self.maximum, values.max())

    def header(self):
        return (np.float32(self.minimum), np.float32(self.maximum),
                np.float32(self.mean), np.float32(np.sqrt(self.squares / self.count)))


def read_defocus_file(format="ctffind4"):
    #TODO
    pass
