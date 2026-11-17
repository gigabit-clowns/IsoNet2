import numpy as np

class reform3D:
    def __init__(self,data3D, cubesize, cropsize, edge_depth):
        self._sp = np.array(data3D.shape)
        self._orig_data = data3D
        self.cubesize = cubesize
        self.cropsize = cropsize
        self.edge_depth = edge_depth
        self._sidelen = np.ceil((self._sp + edge_depth * 2)/self.cubesize).astype(int)
        #self._sidelen = np.ceil((1.*self._sp)/self.cubesize).astype(int)

    def pad_and_crop(self):
        
        #----------------------------|---------------------------
        #|                           |
        #|     ---------------|------|--------edge---------------
        #|     |  ------------|------|--------image_edge----------  
        #|     |  |           |      |
        #|     |  |           |      |
        #|     |  |           |      |
        #|     |  |           |      |
        #|     -----cube-------      | 
        #|     |  |                  |
        #|     |  |                  |
        #|----------crop--------------       
        #|     |  |        
        #|     |  |       
        #|     |  |       
        #|     |  |
        pad_left = int((self.cropsize - self.cubesize)/2 + self.edge_depth)
        
        # pad_right + pad_left + shape = sidelen * cube_zize + (crop_size-cube_size)
        # pad_right + pad_left + shape >= (self._sp + edge_depth * 2) + (crop_size-cube_size)
        pad_right = (self._sidelen * self.cubesize + (self.cropsize-self.cubesize) - pad_left - self._sp).astype(int)

        data = np.pad(self._orig_data,((pad_left,pad_right[0]),(pad_left,pad_right[1]),(pad_left,pad_right[2])),'symmetric').astype(np.float32)
        outdata=[]

        for i in range(self._sidelen[0]):
            for j in range(self._sidelen[1]):
                for k in range(self._sidelen[2]):
                    cube = data[i*self.cubesize:i*self.cubesize+self.cropsize,
                            j*self.cubesize:j*self.cubesize+self.cropsize,
                            k*self.cubesize:k*self.cubesize+self.cropsize]
                    outdata.append(cube)
        outdata=np.array(outdata, dtype = np.float32)
        return outdata
    
    def mask(self, x_len, y_len, z_len):
        # need to consider should partisioned to len+1 so that left and right can add to one
        p = 2*self.edge_depth#(self.cropsize - self.cubesize)
        assert x_len > 2*p
        assert y_len > 2*p
        assert z_len > 2*p

        array_x = np.minimum(np.arange(x_len+1), p) / p
        array_x = array_x * np.flip(array_x)
        array_x  = array_x[np.newaxis,np.newaxis,:]

        array_y = np.minimum(np.arange(y_len+1), p) / p
        array_y = array_y * np.flip(array_y)
        array_y  = array_y[np.newaxis,:,np.newaxis]

        array_z = np.minimum(np.arange(z_len+1), p) / p
        array_z = array_z * np.flip(array_z)
        array_z  = array_z[:,np.newaxis,np.newaxis]

        out = array_x * array_y * array_z
        return out[:x_len,:y_len,:z_len].astype(np.float32)


    def restore(self,cubes, already_cropped=False):

        if not already_cropped:
            start = (self.cropsize-self.cubesize)//2-self.edge_depth
            end = (self.cropsize-self.cubesize)//2+self.cubesize+self.edge_depth
            cubes = cubes[:,start:end,start:end,start:end]

        restored = np.zeros((self._sidelen[0]*self.cubesize+self.edge_depth*2,
                        self._sidelen[1]*self.cubesize+self.edge_depth*2,
                        self._sidelen[2]*self.cubesize+self.edge_depth*2), dtype = np.float32)
        # logging.info("size restored", restored.shape)
        mask_cube = self.mask(self.cubesize+self.edge_depth*2,self.cubesize+self.edge_depth*2,self.cubesize+self.edge_depth*2)
        for i in range(self._sidelen[0]):
            for j in range(self._sidelen[1]):
                for k in range(self._sidelen[2]):
                    restored[i*self.cubesize:(i+1)*self.cubesize+self.edge_depth*2,
                        j*self.cubesize:(j+1)*self.cubesize+self.edge_depth*2,
                        k*self.cubesize:(k+1)*self.cubesize+self.edge_depth*2] \
                        += cubes[i*self._sidelen[1]*self._sidelen[2]+j*self._sidelen[2]+k]\
                            *mask_cube
                        
                    
        p =self.edge_depth*2 #int((self.cropsize-self.cubesize)/2+self.edge_depth)
        restored = restored[p:p+self._sp[0],p:p+self._sp[1],p:p+self._sp[2]]
        return restored

    def mask_old(self):
        from functools import reduce
        c = self.cropsize
        p = (self.cropsize - self.cubesize)
        mask = np.ones((c, c, c))
        f = lambda x: min(x, p)/p 
        for i in range(c):
            for j in range(c):
                for k in range(c):
                    d = [i, c-i, j, c-j, k, c-k]
                    d = map(f,d)
                    d = reduce(lambda a,b: a*b, d)
                    mask[i,j,k] = d
        return mask
    def restore_from_cubes(self,cubes):

        new = np.zeros((self._sidelen[0]*self.cubesize,
                        self._sidelen[1]*self.cubesize,
                        self._sidelen[2]*self.cubesize), dtype = np.float32)
        start=int((self.cropsize-self.cubesize)/2)
        end=int((self.cropsize+self.cubesize)/2)
        
        for i in range(self._sidelen[0]):
            for j in range(self._sidelen[1]):
                for k in range(self._sidelen[2]):
                    new[i*self.cubesize:(i+1)*self.cubesize,
                        j*self.cubesize:(j+1)*self.cubesize,
                        k*self.cubesize:(k+1)*self.cubesize] \
                        = cubes[i*self._sidelen[1]*self._sidelen[2]+j*self._sidelen[2]+k][start:end,start:end,start:end]
        return new[0:self._sp[0],0:self._sp[1],0:self._sp[2]]
    # def pad4times(self,time=4):
    #     sp = np.array(self._orig_data.shape)
    #     sp = np.expand_dims(sp,axis=0)
    #     padsize = (sp // time + 1) * time - sp
    #     self._padsize =padsize
    #     logging.info(padsize, np.zeros((len(self._orig_data.shape),1)))
    #     width = np.concatenate((np.zeros((len(self._orig_data.shape),1),int),padsize.T),axis=1)
    #     return np.pad(self._orig_data,width,'edge')
    def cropback(self,padded):
        sp = padded.shape
        ps = self._padsize
        orig_sp = sp - ps
        print ('orig_sp',orig_sp)
        return padded[:orig_sp[0][0]][:orig_sp[0][1]][:orig_sp[0][2]]

if __name__ == '__main__':
    pass


class FileTiles:
    """The tiles of a tomogram as reform3D cuts them, read from its file a few at a
    time rather than all at once out of a padded copy of it, and blended back as
    reform3D.restore blends them.

    reform3D pads the tomogram symmetrically and cuts crops of `cropsize` every
    `cubesize`. Every value of such a crop is a voxel of the tomogram, so each crop is
    read as a window of `cropsize` inside it, and the crops that reach into the padding
    gather their voxels from that window by the index the padding reflects them to.
    """

    def __init__(self, path, cubesize, cropsize, edge_depth):
        import vitrio as vio

        self.path = path
        self.cubesize = cubesize
        self.cropsize = cropsize
        self.edge_depth = edge_depth
        self.shape = np.array(vio.query_descriptor(path).extents, dtype=int)
        if self.shape.size != 3 or np.any(self.shape < cropsize):
            raise ValueError(
                f"{path} is of shape {tuple(self.shape)}, and its tiles are crops of "
                f"{cropsize} along each axis."
            )
        # reform3D's geometry, over a volume of this shape that holds nothing.
        self._reform = reform3D(
            np.broadcast_to(np.float32(0), tuple(self.shape)), cubesize, cropsize, edge_depth
        )
        self._sidelen = self._reform._sidelen
        self._pad_left = int((cropsize - cubesize) / 2 + edge_depth)

        # Per axis: the window each crop is read from, and the index within it of each
        # voxel of the crop, through the symmetric padding.
        self._windows, self._gathers = [], []
        for n, count in zip(self.shape, self._sidelen):
            starts = np.arange(count) * cubesize - self._pad_left
            sources = starts[:, None] + np.arange(cropsize)
            sources = np.where(sources < 0, -sources - 1, sources)
            sources = np.where(sources >= n, 2 * n - sources - 1, sources)
            windows = np.clip(sources.min(axis=1), 0, n - cropsize)
            self._windows.append(windows)
            self._gathers.append(sources - windows[:, None])
        self._masks = self._reform.mask(*(cubesize + edge_depth * 2,) * 3)

    def __len__(self):
        return int(np.prod(self._sidelen))

    @property
    def row_length(self):
        """How many tiles a row of them along x holds, one after another in `dataset`."""
        return int(self._sidelen[2])

    def _grid(self):
        return np.unravel_index(np.arange(len(self)), tuple(self._sidelen))

    def dataset(self, data_type=None):
        """The tiles as rows of vitrio.data, in the order reform3D cuts them."""
        import vitrio as vio

        grid = self._grid()
        centres = np.stack(
            [windows[i] + self.cropsize // 2 for windows, i in zip(self._windows, grid)], axis=1
        )
        return vio.data.Dataset({
            "window": vio.data.PatchSource(
                [self.path] * len(self), centres, (self.cropsize,) * 3,
                data_type=data_type, check=False,
            ),
            **{axis: gather[i] for axis, gather, i in zip("zyx", self._gathers, grid)},
        })

    def crops(self, batch):
        """The crops of a batch of rows of `dataset`, padding included, as arrays or
        as tensors on the device of the batch, whichever the batch is of."""
        window = batch["window"]
        if isinstance(window, np.ndarray):
            rows = np.arange(len(window))[:, None, None, None]
        else:
            import torch
            rows = torch.arange(len(window), device=window.device)[:, None, None, None]
        z = batch["z"][:, :, None, None]
        y = batch["y"][:, None, :, None]
        x = batch["x"][:, None, None, :]
        return window[rows, z, y, x]

    def blender(self):
        """Where predicted cubes are blended back, as reform3D.restore blends them, a
        layer of tiles along z at a time."""
        return _Blender(self)


class _Blender:
    """Blends the predicted cubes of the tiles back as reform3D.restore does, holding
    only the layer of tiles along z under way and the edge it shares with the next.

    The cubes have to come in the order the tiles are cut. Once a layer is complete,
    no later cube reaches the sections before the next layer starts, and those
    sections of the result are handed out; each voxel sums the same terms in the same
    order as in the whole restored volume.
    """

    def __init__(self, tiles):
        c, e = tiles.cubesize, tiles.edge_depth
        self.tiles = tiles
        self.added = 0
        self._per_layer = int(np.prod(tiles._sidelen[1:]))
        self._layer = np.zeros((c + e * 2, *(tiles._sidelen[1:] * c + e * 2)), dtype=np.float32)

    def add(self, tile, cube):
        """Add the cropped prediction of tile number `tile` where it lies. Returns the
        sections of the result the tile completes, as the index of the first of them
        and their values, or None."""
        c, e = self.tiles.cubesize, self.tiles.edge_depth
        i, j, k = np.unravel_index(tile, tuple(self.tiles._sidelen))
        if tile != self.added:
            raise ValueError(f"tile {tile} of {self.tiles.path} came after {self.added - 1}.")
        self._layer[:, j*c:(j+1)*c+e*2, k*c:(k+1)*c+e*2] += cube * self.tiles._masks
        self.added += 1
        if self.added % self._per_layer:
            return None

        # The layer starts at section i*c of the restored volume; the result starts
        # at section p of it.
        last = i == self.tiles._sidelen[0] - 1
        done = len(self._layer) if last else c
        p, shape = e * 2, self.tiles.shape
        first, end = max(i * c, p), min(i * c + done, p + shape[0])
        sections = self._layer[first - i*c:end - i*c, p:p+shape[1], p:p+shape[2]].copy()
        if not last:
            following = np.zeros_like(self._layer)
            following[:e * 2] = self._layer[c:]
            self._layer = following
        return (first - p, sections) if end > first else None

    def check_complete(self):
        if self.added != len(self.tiles):
            raise RuntimeError(
                f"{self.added} of the {len(self.tiles)} tiles of {self.tiles.path} were blended."
            )
