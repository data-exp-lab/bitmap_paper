"""
Plots for Bitmap Indexing paper.



"""

import numpy as np
import time
import sys
import os
import copy
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as optimization

sys.path.insert(0, "/root/python/yt-bitmap/")
import yt
print("Using yt installed in {}".format(os.path.dirname(yt.__file__)))

from yt.geometry.particle_oct_container import \
    ParticleBitmap
from yt.geometry.oct_container import _ORDER_MAX
from yt.geometry.selection_routines import RegionSelector, AlwaysSelector
from yt.testing import \
    assert_equal, \
    requires_file, \
    assert_true, \
    assert_array_equal
from yt.units.unit_registry import UnitRegistry
from yt.units.yt_array import YTArray
from yt.utilities.lib.geometry_utils import \
    get_morton_indices, \
    get_morton_points, \
    get_hilbert_points, \
    get_hilbert_indices
from yt.funcs import get_pbar

import yt.units.dimensions as dimensions

class FakeDS:
    r"""Fake empty dataset.

    Attributes:
        domain_left_edge (YTArray): Left edges of domain.
        domain_right_edge (YTArray): Right edges of domain.
        domain_width (YTArray): Domain width in each dimension.
        unit_registry (UnitRegistry): Domain units.
        periodicity (tuple of bool): Periodicity of each dimension.

    """
    domain_left_edge = None
    domain_right_edge = None
    domain_width = None
    unit_registry = UnitRegistry()
    unit_registry.add('code_length', 1.0, dimensions.length)
    periodicity = (False, False, False)

class FakeBoxRegion:
    r"""Fake box-shaped region.

    Args:
        nfiles (int): Number of files dataset is split accross.
        DLE (array-like): Left edges of domain.
        DRE (array-like): Right edges of domain.

    Attributes:
        ds (FakeDS): Fake dataset that region intersects.
        nfiles (int): Number of files dataset is split accross.
        left_edge (YTArray): Left edges of the box shaped region.
        right_edge (YTArray): Right edges of the box shaped region.

    """
    def __init__(self, nfiles, DLE, DRE):
        self.ds = FakeDS()
        self.ds.domain_left_edge = YTArray(DLE, "code_length",
                                           registry=self.ds.unit_registry)
        self.ds.domain_right_edge = YTArray(DRE, "code_length",
                                            registry=self.ds.unit_registry)
        self.ds.domain_width = self.ds.domain_right_edge - \
                               self.ds.domain_left_edge
        self.nfiles = nfiles
        self.left_edge = None
        self.right_edge = None

    def set_edges(self, center, width):
        r"""Set the edges of the box-shaed region.

        Args:
            center (np.ndarray of float64): Position of the box center relative 
                to the domain left edge (i.e. [0.5,0.5,0.5] would be the domain 
                center.
            width (np.ndarray of flaot64): Width of the box in each dimension, 
                relative to the domain width.

        """
        self.left_edge = self.ds.domain_left_edge + self.ds.domain_width*(center-width/2)
        self.right_edge = self.ds.domain_left_edge + self.ds.domain_width*(center+width/2)


def fake_decomp_random(npart, nfiles, ifile, DLE, DRE,
                       buff=0.0, verbose=False):
    r"""Generate points for one file in a fake dataset where the points are 
    partitioned between the files randomly.

    Args:
        npart (int): Total number of points in the dataset.
        nfiles (int): Total number of files the dataset is split accross.
        ifile (int): The index of the file in question.
        DLE (YTArray): Left edges of the domain.
        DRE (YTArray): Right edges of the domain.
        buff (float, optional): Not used.
        verbose (bool, optional): If True, additional information is printed 
            about this file. Defaults to False.

    Returns:
        np.ndarray: Coordinates of points in the ith file from a fake dataset.

    """
    np.random.seed(int(0x4d3d3d3)+ifile)
    DW = DRE - DLE
    nPF = int(npart/nfiles)
    nR = npart % nfiles
    if verbose: print("{}/{} remainder particles put in first file".format(nR,npart))
    if ifile == 0:
        pos = np.random.normal(0.5, scale=0.05, size=(nPF+nR,3))*DW + DLE
    else:
        pos = np.random.normal(0.5, scale=0.05, size=(nPF,3))*DW + DLE
    for i in range(3):
        np.clip(pos[:,i], DLE[i], DRE[i], pos[:,i])
    return pos

def fake_decomp_sliced(npart, nfiles, ifile, DLE, DRE,
                       buff=0.0, verbose=False):
    r"""Generate points for one file in a fake dataset where the points are 
    partitioned between the files in slices along one dimension.

    Args:
        npart (int): Total number of points in the dataset.
        nfiles (int): Total number of files the dataset is split accross.
        ifile (int): The index of the file in question.
        DLE (YTArray): Left edges of the domain.
        DRE (YTArray): Right edges of the domain.
        buff (float, optional): Fractional overlap between points in neighboring 
            domain slices. Defaults to 0.0 (no overlap).
        verbose (bool, optional): If True, additional information is printed 
            about this file. Defaults to False.

    Returns:
        np.ndarray: Coordinates of points in the ith file from a fake dataset.

    """
    np.random.seed(int(0x4d3d3d3)+ifile)
    DW = DRE - DLE
    div = DW/nfiles
    nPF = int(npart/nfiles)
    nR = npart % nfiles
    if verbose: print("{}/{} remainder particles put in first file".format(nR,npart))
    inp = nPF
    if ifile == 0: inp += nR
    iLE = DLE[0] + ifile*div[0]
    iRE = iLE + div[0]
    if ifile != 0:
        iLE -= buff*div[0]
    if ifile != (nfiles-1):
        iRE += buff*div[0]
    pos = np.empty((inp,3), dtype='float')
    pos[:,0] = np.random.uniform(iLE, iRE, inp)
    for i in range(1,3):
        pos[:,i] = np.random.uniform(DLE[i], DRE[i], inp)
    return pos


def makeall_decomp_hilbert_gaussian(npart, nfiles, DLE, DRE,
                                    buff=0.0, order=6, verbose=False,
                                    fname_base=None, nchunk=10,
                                    width=None, center=None,
                                    frac_random=0.1):
    r"""Create all files in a fake dataset where the points are paritioned 
    between the files according to a Hilbert space filling curve. The points 
    are distributed according to a gaussian except for a fraction that are 
    randomly distributed in order to produce overlap between files.

    Args:
        npart (int): Total number of points in the dataset.
        nfiles (int): Total number of files the dataset is split accross.
        DLE (YTArray): Left edges of the domain.
        DRE (YTArray): Right edges of the domain.
        buff (float, optional): Fractional overlap between points in neighboring 
            domain slices. Defaults to 0.0.
        order (int, optional): Order of Hilbert space filling curve used for 
            domain decomposition between files. Defaults to 6.
        verbose (bool, optional): If True, additional information is printed 
            about each file. Defaults to False.
        fname_base (str, optional): Base name for each file in the fake dataset 
            being generated. Defaults to None and is set based on input to be:
            `'hilbert{}_gaussian_np{}_nf{}_'.format(order,npart,nfiles)`
        nchunk (int, optional): Number of chunks to split each file between in 
            order to limit memory. Defaults to 10.
        width (np.ndarray of float, optional): Width of point distribution 
            gaussian in each dimension. Defaults to `0.1*(DRE - DLE)`.
        center (np.ndarray of flaot, optional): Center of point distribution
            in each dimension. Defaults to `DLE + 0.5*(DRE - DLE)`.
        frac_random (float, optional): Fraction of the total particles that 
            are randomly distributed. Defaults to 0.1.

    """
    np.random.seed(int(0x4d3d3d3))
    DW = DRE - DLE
    if fname_base is None:
        fname_base = 'hilbert{}_gaussian_np{}_nf{}_'.format(order,npart,nfiles)
    if width is None:
        width = 0.1*DW
    if center is None:
        center = DLE+0.5*DW
    def load_pos(file_id):
        filename = fname_base+'file{}'.format(file_id)
        if os.path.isfile(filename):
            fd = open(filename,'rb')
            positions = pickle.load(fd)
            fd.close()
        else:
            positions = np.empty((0,3), dtype='float64')
        return positions
    def save_pos(file_id,positions):
        filename = fname_base+'file{}'.format(file_id)
        fd = open(filename,'wb')
        pickle.dump(positions,fd)
        fd.close()
    npart_rnd = int(frac_random*npart)
    npart_gau = npart - npart_rnd
    dim_hilbert = (1<<order)
    nH = dim_hilbert**3
    if nH < nfiles:
        raise ValueError('Fewer hilbert cells than files.')
    nHPF = nH/nfiles
    rHPF = nH%nfiles
    hdiv = DW/dim_hilbert
    for ichunk in range(nchunk):
        print "Chunk {}...".format(ichunk)
        inp = npart_gau/nchunk
        if ichunk == 0: inp += (npart_gau % nchunk)
        pos = np.empty((inp,3), dtype='float64')
        ind = np.empty((inp,3), dtype='int64')
        for k in range(3):
            pos[:,k] = np.clip(np.random.normal(center[k], width[k], inp),
                               DLE[k], DRE[k]-(1.0e-9)*DW[k])
            ind[:,k] = (pos[:,k]-DLE[k])/(DW[k]/dim_hilbert)
        harr = get_hilbert_indices(order, ind)
        farr = (harr-rHPF)/nHPF
        for ifile in range(nfiles):
            print "Chunk {}, file {}...".format(ichunk,ifile)
            ipos = load_pos(ifile)
            if ifile == 0:
                idx = (farr <= ifile) # Put remainders in first file
            else:
                idx = (farr == ifile)
            ipos = np.concatenate((ipos,pos[idx,:]),axis=0)
            save_pos(ifile,ipos)
    # Random
    print "Random..."
    for ifile in range(nfiles):
        # print 'Random, file {}'.format(ifile)
        ipos = load_pos(ifile)
        ipos_rnd = fake_decomp_hilbert_uniform(npart_rnd, nfiles, ifile, DLE, DRE,
                                               buff=buff, order=order, verbose=verbose)
        ipos = np.concatenate((ipos,ipos_rnd),axis=0)
        save_pos(ifile,ipos)
    
def fake_decomp_hilbert_gaussian(npart, nfiles, ifile, DLE, DRE,
                                 buff=0.0, order=6, verbose=False):
    r"""Generate points for one file in a fake dataset where the points are 
    partitioned between the files according to a Hilbert space filling curve.
    The points are distributed throughout the domain according to a gaussian.

    Args:
        npart (int): Total number of points in the dataset.
        nfiles (int): Total number of files the dataset is split accross.
        ifile (int): The index of the file in question.
        DLE (YTArray): Left edges of the domain.
        DRE (YTArray): Right edges of the domain.
        buff (float, optional): Fractional overlap between points in neighboring 
            domain slices. Defaults to 0.0 (no overlap).
        order (int, optional): Order of Hilbert space filling curve used for 
            domain decomposition between files. Defaults to 6.
        verbose (bool, optional): If True, additional information is printed 
            about this file. Defaults to False.

    Returns:
        np.ndarray: Coordinates of points in the ith file from a fake dataset.

    """
    np.random.seed(int(0x4d3d3d3))
    DW = DRE - DLE
    dim_hilbert = (1<<order)
    nH = dim_hilbert**3
    if nH < nfiles:
        raise Exception('Fewer hilbert cells than files.')
    nHPF = nH/nfiles
    rHPF = nH%nfiles
    hdiv = DW/dim_hilbert
    if ifile == 0:
        hlist = np.arange(0,nHPF+rHPF, dtype='int64')
    else:
        hlist = np.arange(ifile*nHPF+rHPF,(ifile+1)*nHPF+rHPF, dtype='int64')
    hpos = get_hilbert_points(order, hlist)
    iLE = np.empty((len(hlist),3), dtype='float')
    iRE = np.empty((len(hlist),3), dtype='float')
    count = np.zeros(3,dtype='int')
    pos = np.empty((npart,3), dtype='float')
    for k in range(3):
        iLE[:,k] = DLE[k] + hdiv[k]*hpos[:,k]
        iRE[:,k] = iLE[:,k] + hdiv[k]
        iLE[hpos[:,k]!=0,k] -= buff*hdiv[k]
        iRE[hpos[:,k]!=(dim_hilbert-1),k] += buff*hdiv[k]
        print 'sampling'
        gpos = np.clip(np.random.normal(DLE[k]+DW[k]/2.0, DW[k]/10.0, npart),
                       DLE[k], DRE[k])
        print 'sampled'
        for p,ipos in enumerate(gpos):
            if (p%(10**8))==0: print "    dim {}, part {} ({})".format(k,p,ipos)
            for i in range(len(hlist)):
                if iLE[i,k] <= ipos < iRE[i,k]:
                    pos[count[k],k] = ipos
                    count[k] += 1
                    break
    return pos[:count.min(),:]
    
def fake_decomp_hilbert_uniform(npart, nfiles, ifile, DLE, DRE,
                                buff=0.0, order=6, verbose=False):
    r"""Generate points for one file in a fake dataset where the points are 
    partitioned between the files according to a Hilbert space filling curve.
    The points are distributed throughout the domain uniformly.

    Args:
        npart (int): Total number of points in the dataset.
        nfiles (int): Total number of files the dataset is split accross.
        ifile (int): The index of the file in question.
        DLE (YTArray): Left edges of the domain.
        DRE (YTArray): Right edges of the domain.
        buff (float, optional): Fractional overlap between points in neighboring 
            domain slices. Defaults to 0.0 (no overlap).
        order (int, optional): Order of Hilbert space filling curve used for 
            domain decomposition between files. Defaults to 6.
        verbose (bool, optional): If True, additional information is printed 
            about this file. Defaults to False.

    Returns:
        np.ndarray: Coordinates of points in the ith file from a fake dataset.

    """
    np.random.seed(int(0x4d3d3d3)+ifile)
    DW = DRE - DLE
    dim_hilbert = (1<<order)
    nH = dim_hilbert**3
    if nH < nfiles:
        raise Exception('Fewer hilbert cells than files.')
    nHPF = nH/nfiles
    rHPF = nH%nfiles
    nPH = npart/nH
    nRH = npart%nH
    hind = np.arange(nH, dtype='int64')
    hpos = get_hilbert_points(order, hind)
    hdiv = DW/dim_hilbert
    if ifile == 0:
        hlist = range(0,nHPF+rHPF)
        nptot = nPH*len(hlist)+nRH
    else:
        hlist = range(ifile*nHPF+rHPF,(ifile+1)*nHPF+rHPF)
        nptot = nPH*len(hlist)
    pos = np.empty((nptot,3), dtype='float')
    pc = 0
    for i in hlist:
        iLE = DLE + hdiv*hpos[i,:]
        iRE = iLE + hdiv
        for k in range(3): # Don't add buffer past domain bounds
            if hpos[i,k] != 0:
                iLE[k] -= buff*hdiv[k]
            if hpos[i,k] != (dim_hilbert-1):
                iRE[k] += buff*hdiv[k]
        inp = nPH
        if (ifile == 0) and (i == 0): inp += nRH
        for k in range(3):
            pos[pc:(pc+inp),k] = np.random.uniform(iLE[k], iRE[k], inp)
        pc += inp
    return pos

def fake_decomp_morton(npart, nfiles, ifile, DLE, DRE,
                        buff=0.0, order=6, verbose=False):
    r"""Generate points for one file in a fake dataset where the points are 
    partitioned between the files according to a Morton space filling curve.
    The points are distributed throughout the domain uniformly.

    Args:
        npart (int): Total number of points in the dataset.
        nfiles (int): Total number of files the dataset is split accross.
        ifile (int): The index of the file in question.
        DLE (YTArray): Left edges of the domain.
        DRE (YTArray): Right edges of the domain.
        buff (float, optional): Fractional overlap between points in neighboring 
            domain slices. Defaults to 0.0 (no overlap).
        order (int, optional): Order of Morton space filling curve used for 
            domain decomposition between files. Defaults to 6.
        verbose (bool, optional): If True, additional information is printed 
            about this file. Defaults to False.

    Returns:
        np.ndarray: Coordinates of points in the ith file from a fake dataset.

    """
    np.random.seed(int(0x4d3d3d3)+ifile)
    DW = DRE - DLE
    dim_morton = (1<<order)
    nH = dim_morton**3
    if nH < nfiles:
        raise Exception('Fewer morton cells than files.')
    nHPF = nH/nfiles
    rHPF = nH%nfiles
    nPH = npart/nH
    nRH = npart%nH
    hind = np.arange(nH, dtype='uint64')
    hpos = get_morton_points(hind)
    hdiv = DW/dim_morton
    if ifile == 0:
        hlist = range(0,nHPF+rHPF)
        nptot = nPH*len(hlist)+nRH
    else:
        hlist = range(ifile*nHPF+rHPF,(ifile+1)*nHPF+rHPF)
        nptot = nPH*len(hlist)
    pos = np.empty((nptot,3), dtype='float')
    pc = 0
    for i in hlist:
        iLE = DLE + hdiv*hpos[i,:]
        iRE = iLE + hdiv
        for k in range(3): # Don't add buffer past domain bounds
            if hpos[i,k] != 0:
                iLE[k] -= buff*hdiv[k]
            if hpos[i,k] != (dim_morton-1):
                iRE[k] += buff*hdiv[k]
        inp = nPH
        if (ifile == 0) and (i == 0): inp += nRH
        for k in range(3):
            pos[pc:(pc+inp),k] = np.random.uniform(iLE[k], iRE[k], inp)
        pc += inp
    return pos

def fake_decomp_grid(npart, nfiles, ifile, DLE, DRE, verbose=False):
    r"""Generate points for one file in a fake dataset where the points are 
    partitioned between the files according to a grid. The points are 
    distributed throughout the domain uniformly.

    Args:
        npart (int): Total number of points in the dataset.
        nfiles (int): Total number of files the dataset is split accross.
        ifile (int): The index of the file in question.
        DLE (YTArray): Left edges of the domain.
        DRE (YTArray): Right edges of the domain.
        buff (float, optional): Fractional overlap between points in neighboring 
            domain slices. Defaults to 0.0 (no overlap).
        verbose (bool, optional): If True, additional information is printed 
            about this file. Defaults to False.

    Returns:
        np.ndarray: Coordinates of points in the ith file from a fake dataset.

    """
    # TODO: handle 'remainder' particles
    np.random.seed(int(0x4d3d3d3)+ifile)
    DW = DRE - DLE
    nYZ = int(np.sqrt(npart/nfiles))
    nR = npart - nYZ*nYZ*nfiles
    div = DW/nYZ
    Y, Z = np.mgrid[DLE[1] + 0.1*div[1] : DRE[1] - 0.1*div[1] : nYZ * 1j,
                    DLE[2] + 0.1*div[2] : DRE[2] - 0.1*div[2] : nYZ * 1j]
    X = 0.5 * div[0] * np.ones(Y.shape, dtype="float64") + div[0]*ifile
    pos = np.array([X.ravel(),Y.ravel(),Z.ravel()],
                   dtype="float64").transpose()
    return pos

def yield_fake_decomp(decomp, npart, nfiles, DLE, DRE, **kws):
    r"""Yield points for each file in a fake dataset where the points are 
    partitioned between the files according to some domain domain decomposition.

    Args:
        decomp (str): Name of domain decomposition scheme that should be used to
            select points for this file. Valid values include:
              'grid': A perfect grid with no overlap.
              'random': Random points from entire domain.
              'sliced': Points from a slice in one dimension.
              'hilbert': Points from consecutive cells along a Hilbert space
                  filling curve.
              'morton': Points from consecutive cells along a Morton space 
                  filling curve.
            If `decomp.startswith('zoom_')`, then the dataset returned is the 
            will have half the particles distributed across the whole domain and 
            half distributed across one fifth of the domain (at the center), but 
            using scaled domain decompositions. Multiple domain decompositions 
            can be combined by joining them with an underscore. Then the points 
            are split evenly between each scheme.
        npart (int): Total number of points in the dataset.
        nfiles (int): Total number of files the dataset is split accross.
        DLE (YTArray): Left edges of the domain.
        DRE (YTArray): Right edges of the domain.
        **kwargs: Additional keyword arguments are passed to :meth:`fake_decomp`.

    Yields:
        np.ndarray: Coordinates of points in the ith file from a fake dataset.

    """
    for ifile in range(nfiles):
        yield fake_decomp(decomp, npart, nfiles, ifile, DLE, DRE, **kws)

def fake_decomp(decomp, npart, nfiles, ifile, DLE, DRE, 
                distrib='uniform', fname=None, **kws):
    r"""Generate points for one file in a fake dataset where the points are 
    partitioned between the files according to some domain domain decomposition.

    Args:
        decomp (str): Name of domain decomposition scheme that should be used to
            select points for this file. Valid values include:
              'grid': A perfect grid with no overlap.
              'random': Random points from entire domain.
              'sliced': Points from a slice in one dimension.
              'hilbert': Points from consecutive cells along a Hilbert space
                  filling curve.
              'morton': Points from consecutive cells along a Morton space 
                  filling curve.
            If `decomp.startswith('zoom_')`, then the dataset returned is the 
            will have half the particles distributed across the whole domain and 
            half distributed across one fifth of the domain (at the center), but 
            using scaled domain decompositions. Multiple domain decompositions 
            can be combined by joining them with an underscore. Then the points 
            are split evenly between each scheme.
        npart (int): Total number of points in the dataset.
        nfiles (int): Total number of files the dataset is split accross.
        ifile (int): The index of the file in question.
        DLE (YTArray): Left edges of the domain.
        DRE (YTArray): Right edges of the domain.
        distrib (str, optional): Description of how points should be distributed 
            across the domain. Defaults to 'uniform'. Valid values include:
              'uniform': Uniform distribution throughout the domain.
              'gaussian': Distribution according to gaussian. Only valid for 
                  `decomp == 'hilbert'`.
        fname (str, optional): Name of file where points should be loaded from/
            saved to. Defaults to None. If None, points are not saved.
        **kwargs: Additional keyword arguments are passed to the individual
            domain decompositon methods.

    Returns:
        np.ndarray: Coordinates of points in the ith file from a fake dataset.

    Raises:
        ValueError: If `decomp` is not in the list of accepted values.
        ValueError: If `distrib` is not in the list of accepted values for the 
            given value of `decomp`.

    """
    if fname is None and distrib == 'gaussian':
        fname = '{}6_{}_np{}_nf{}_file{}'.format(decomp,distrib,npart,nfiles,ifile)
    if fname is not None and os.path.isfile(fname):
        fd = open(fname,'rb')
        pos = pickle.load(fd)
        fd.close()
        return pos
    if decomp.startswith('zoom_'):
        zoom_factor = 5
        decomp_zoom = decomp.split('zoom_')[-1]
        zoom_npart = npart/2
        zoom_rem = npart%2
        pos1 = fake_decomp(decomp_zoom, zoom_npart+zoom_rem, 
                           nfiles, ifile, DLE, DRE, distrib=distrib, **kws)
        DLE_zoom = DLE + 0.5*DW*(1.0 - 1.0/float(zoom_factor))
        DRE_zoom = DLE_zoom + DW/zoom_factor
        pos2 = fake_decomp(decomp_zoom, zoom_npart, nfiles, ifile,
                                  DLE_zoom, DRE_zoom, distrib=distrib, **kws)
        pos = np.concatenate((pos1,pos2),axis=0)
    elif '_' in decomp:
        decomp_list = decomp.split('_')
        decomp_np = npart/len(decomp_list)
        decomp_nr = npart%len(decomp_list)
        pos = np.empty((0,3), dtype='float')
        for i,idecomp in enumerate(decomp_list):
            inp = decomp_np
            if i == 0:
                inp += decomp_nr
            ipos = fake_decomp(idecomp, inp, nfiles, ifile, DLE, DRE, 
                               distrib=distrib, **kws)
            pos = np.concatenate((pos,ipos),axis=0)
    # A perfect grid, no overlap between files
    elif decomp == 'grid':
        buff = kws.pop('buff',None)
        pos = fake_decomp_grid(npart, nfiles, ifile, DLE, DRE, **kws)
    # Completely random data set
    elif decomp == 'random':
        if distrib == 'uniform':
            pos = fake_decomp_random(npart, nfiles, ifile, DLE, DRE, **kws)
        else:
            raise ValueError("Unsupported value for input parameter 'distrib'".format(distrib))
    # Each file contains a slab (part of x domain, all of y/z domain)
    elif decomp == 'sliced':
        if distrib == 'uniform':
            pos = fake_decomp_sliced(npart, nfiles, ifile, DLE, DRE, **kws)
        else:
            raise ValueError("Unsupported value for input parameter 'distrib'".format(distrib))
    # Particles are assigned to files based on their location on a
    # Peano-Hilbert curve of order 6
    elif decomp.startswith('hilbert'):
        if decomp == 'hilbert':
            kws['order'] = 6
        else:
            kws['order'] = int(decomp.split('hilbert')[-1])
        if distrib == 'uniform':
            pos = fake_decomp_hilbert_uniform(npart, nfiles, ifile, DLE, DRE, **kws)
        elif distrib == 'gaussian':
            makeall_decomp_hilbert_gaussian(npart, nfiles, DLE, DRE, 
                                            fname_base=fname.split('file')[0], **kws)
            pos = fake_decomp(decomp, npart, nfiles, ifile, DLE, DRE,
                              distrib=distrib, fname=fname, **kws)
            # pos = fake_decomp_hilbert_gaussian(npart, nfiles, ifile, DLE, DRE, **kws)
        else:
            raise ValueError("Unsupported value for input parameter 'distrib'".format(distrib))
    # Particles are assigned to files based on their location on a
    # Morton ordered Z-curve of order 6
    elif decomp.startswith('morton'):
        if decomp == 'morton':
            kws['order'] = 6
        else:
            kws['order'] = int(decomp.split('morton')[-1])
        if distrib == 'uniform':
            pos = fake_decomp_morton(npart, nfiles, ifile, DLE, DRE, **kws)
        else:
            raise ValueError("Unsupported value for input parameter 'distrib'".format(distrib))
    else:
        raise ValueError("Unsupported value {} for input parameter 'decomp'".format(decomp))
    # Save
    if fname is not None:
        fd = open(fname,'wb')
        pickle.dump(pos,fd)
        fd.close()
    return pos

def FakeBitmap(npart, nfiles, order1, order2, decomp='grid', 
               buff=0.5, DLE=None, DRE=None, distrib='uniform',
               fname=None, verbose=False, really_verbose=False):
    r"""Create a bitmap for a fake dataset.

    Args:
        npart (int): Total number of points in the dataset.
        nfiles (int): Total number of files the dataset is split accross.
        order1 (int): Order of the coarse index.
        order2 (int): Order of the refined index.
        decomp (str): Name of domain decomposition scheme that should be used to
            select points for this file. Valid values include:
              'grid': A perfect grid with no overlap.
              'random': Random points from entire domain.
              'sliced': Points from a slice in one dimension.
              'hilbert': Points from consecutive cells along a Hilbert space
                  filling curve.
              'morton': Points from consecutive cells along a Morton space 
                  filling curve.
            If `decomp.startswith('zoom_')`, then the dataset returned is the 
            will have half the particles distributed across the whole domain and 
            half distributed across one fifth of the domain (at the center), but 
            using scaled domain decompositions. Multiple domain decompositions 
            can be combined by joining them with an underscore. Then the points 
            are split evenly between each scheme.
        buff (float, optional): Fractional overlap between points in neighboring 
            domain slices. Defaults to 0.5.
        DLE (YTArray, optional): Left edges of the domain. Defaults to None.
        DRE (YTArray, optional): Right edges of the domain. Defaults to None.
        distrib (str, optional): Description of how points should be distributed 
            across the domain. Defaults to 'uniform'. Valid values include:
              'uniform': Uniform distribution throughout the domain.
              'gaussian': Distribution according to gaussian. Only valid for 
                  `decomp == 'hilbert'`.
        fname (str, optional): Name of file where bitmap should be loaded from/
            saved to. Defaults to None. If None, bitmap is not saved.
        verbose (bool, optional): If True, information about the bitmap 
            construction is printed. Defaults to False.
        really_verbose (bool, optional): If True, information about the 
            construction of each file in the fake dataset is printed. Defaults 
            to False.

    Returns:
        reg (ParticleBitmap): Bitmap object.
        cc (?): File collisions at coarse index.
        rc (?): File collisions at refined index.
        mem (int): Size of bitmasks in bytes.

    """
    N = (1<<order1)
    if DLE is None: DLE = np.array([0.0, 0.0, 0.0])
    if DRE is None: DRE = np.array([1.0, 1.0, 1.0])
    reg = ParticleBitmap(DLE, DRE, nfiles,
                         index_order1 = order1,
                         index_order2 = order2)
    # Load from file if it exists
    if isinstance(fname,str) and os.path.isfile(fname):
        reg.load_bitmasks(fname)
        cc = reg.find_collisions_coarse(verbose=verbose)
        rc = reg.find_collisions_refined(verbose=verbose)
    else:
        # Create positions for each file
        posgen = yield_fake_decomp(decomp, npart, nfiles, DLE, DRE, buff=buff,
                                   distrib=distrib, verbose=really_verbose)
        # Coarse index
        cp = 0
        pb = get_pbar("Initializing coarse index ",nfiles)
        max_npart = 0
        for i,pos in enumerate(posgen):
            pb.update(i)
            reg._coarse_index_data_file(pos, i)
            max_npart = max(max_npart, pos.shape[0])
            cp += pos.shape[0]
        pb.finish()
        if i != (nfiles-1):
            raise RuntimeError("There are positions for {} files, but there should be {}.".format(i+1,nfiles))
        if really_verbose: print("{} particles in total".format(cp))
        cc = reg.find_collisions_coarse(verbose=verbose)
        # Refined index
        sub_mi1 = np.zeros(max_npart, "uint64")
        sub_mi2 = np.zeros(max_npart, "uint64")
        posgen = yield_fake_decomp(decomp, npart, nfiles, DLE, DRE, buff=buff, 
                                   distrib=distrib, verbose=really_verbose)
        pb = get_pbar("Initializing refined index ",nfiles)
        for i,pos in enumerate(posgen):
            pb.update(i)
            reg._refined_index_data_file(pos,
                                         reg.masks.sum(axis=1).astype('uint8'),
                                         sub_mi1, sub_mi2, i)
        pb.finish()
        rc = reg.find_collisions_refined(verbose=verbose)
        # Owners
        reg.set_owners()
        # Save if file name provided
        if isinstance(fname,str):
            reg.save_bitmasks(fname)
    mem = reg.calcsize_bitmasks()
    return reg, cc, rc, mem

def vary_selection_stats(var, varlist, verbose=False, plot=False,
                         nfiles=512, npart_dim=1024, 
                         DLE = [0.0, 0.0, 0.0],
                         DRE = [1.0, 1.0, 1.0], 
                         overwrite=False, extendtag=None, **kws):
    r"""Get info on the performance of bitmap selection on a test dataset for 
    various parameter values.

    Args:
        var (str): Variable that should be varied.
        varlist (list): List of values for the variable identified by `var`.
        verbose (bool, optional): If True, information is printed to the screen.
            Defaults to False.
        plot (bool, optional): If True, generic info is ploted. Defaults to 
            False.
        nfiles (int, optional): Number of files in fake dataset. Defaults to 512.
        npart_dim (int, optional): Number of points per dimension in fake 
            dataset. Defaults to 1024.
        DLE (array-like, optional): Left edges of domain. Defaults to [0,0,0].
        DRE (array-like, optional): Right edges of domain. Defaults to [1,1,1].
        overwrite (bool, optional): If True, results saved to disk are 
            overwritten with new ones. Otherwise, results are loaded form the 
            pre-existing results. Defaults to False.
        extendtag (str, optional): Additional string to append to file names.
            Defaults to None and name is not appended.
        **kws: Additional keyword arguments are passed to :meth:`time_selection`.

    Returns:
        dict: Dictionary where keys are entries from `varlist` and values are 
            dictionaries of performance statistics returned by 
            :meth:`time_selection`.

    """
    kwsDEF = dict(decomp='hilbert',
                  buff=0.1,
                  distrib='uniform',
                  ngz=0,
                  nreps=10)
    for k in kwsDEF: kws.setdefault(k,kwsDEF[k])
    testtag = "vary_{}_np{}_nf{}_{}_buff{}_{}ngz_{}reps".format(var,npart_dim,nfiles,
                                                                kws['decomp'],str(kws['buff']).replace('.','p'),
                                                                kws['ngz'],kws['nreps'])
    if kws['distrib'] != 'uniform':
        testtag += '_{}'.format(kws['distrib'])
    if extendtag is not None:
        testtag += extendtag
    fname = testtag+'.dat'
    # Create regions
    fake_regions = []
    if var == 'selector':
        for v in varlist:
            fr = FakeBoxRegion(nfiles, DLE, DRE)
            fr.set_edges(0.5,v)
            fake_regions.append(fr)
    else:
        for c,r in [(0.5,0.1),(0.3,0.1),(0.5,0.01),(0.5,0.2),(0.5,0.5),(0.5,1.0)]:
            fr = FakeBoxRegion(nfiles, DLE, DRE)
            fr.set_edges(c,r)
            fake_regions.append(fr)
    # Load
    if os.path.isfile(fname) and not overwrite:
        fd = open(fname,'rb')
        out = pickle.load(fd)
        fd.close()
    else:
        out = {}
    # Get stats
    if verbose: print("Timing differences due to '{}'".format(var))
    if var == 'selector':
        iout = time_selection(npart_dim, nfiles, fake_regions, 
                              verbose=verbose, total_regions=False,
                              **copy.copy(kws))
        outkws = iout.keys()
        for i,v in enumerate(varlist):
            out[v] = {}
            for k in outkws: out[v][k] = iout[k][i]
            if verbose: print("{var} = {v}: {tm} s, {ndf}/{nf} files, {ngf}/{nf} ghost files".format(var=var,v=v,**out[v]))
    else:
        for v in varlist:
            if v in out: continue
            kws[var] = v
            out[v] = time_selection(npart_dim, nfiles, fake_regions, 
                                    verbose=verbose, total_regions=True,
                                    **copy.copy(kws))
            if verbose: print("{var} = {v}: {tm} s, {ndf}/{nf} files, {ngf}/{nf} ghost files".format(var=var,v=v,**out[v]))
    # Save
    fd = open(fname,'wb')
    pickle.dump(out,fd)
    fd.close()
    # Plot
    if plot:
        plotfile = os.path.join(os.getcwd(),testtag+'.png')
        plot_vary_selection_stats(var, varlist, out, fname=plotfile)
    return out

def plot_vary_selection_stats(var, varlist, result, fname=None):
    r"""Plot generic performance results.

    Args:
        var (str): Variable that should be varied.
        varlist (list): List of values for the variable identified by `var`.
        result (dict): Results dictionary from :meth:`vary_selection_stats`.
        fname (str, optional): Name of file where plot should be saved. If 
            None, plot is displayed and not saved. Defaults to None.

    """
    Nvar = len(varlist)
    t = np.empty(Nvar, dtype='float')
    df = np.empty(Nvar, dtype='float')
    gf = np.empty(Nvar, dtype='float')
    nf = np.empty(Nvar, dtype='float')
    cc = np.empty(Nvar, dtype='float')
    rc = np.empty(Nvar, dtype='float')
    for i,v in enumerate(varlist):
        t[i] = result[v]['tm']
        df[i] = result[v]['ndf']
        gf[i] = result[v]['ngf']
        nf[i] = result[v]['nf']
        cc[i] = float(result[v]['cc'][0])/float(result[v]['cc'][1])
        rc[i] = float(result[v]['rc'][0])/float(result[v]['rc'][1])
    # Plot
    plt.close('all')
    f, ax1 = plt.subplots()
    if var == 'decomp':
        ax1.scatter(range(Nvar),t,c='k',marker='o',s=50,label='Time')
    elif var == 'selector':
        ax1.semilogx(varlist,t,'k-',label='Time')
    elif var in ['order1','order2']:
        ax1.semilogy(varlist,t,'k-',label='Time')
    else:
        ax1.plot(varlist,t,'k-',label='Time')
    ax1.set_xlabel(var)
    ax1.set_ylabel('Time (s)')
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(4)
    ax1.tick_params(width=4)
    # Files and collitions
    ax2 = ax1.twinx()
    ax2.set_ylabel('% files/collisions')
    if var == 'decomp':
        ax2.scatter(range(Nvar),df/nf,c='b',marker='^',s=50,label='Primary Files')
        ax2.scatter(range(Nvar),gf/nf,c='b',marker='s',s=50,label='Ghost Files')
        ax2.scatter(range(Nvar),cc,c='r',marker='>',s=50,label='Coarse Collisions')
        ax2.scatter(range(Nvar),rc,c='r',marker='<',s=50,label='Refined Collisions')
        xticks = ax2.set_xticklabels(['']+varlist)
        plt.setp(xticks, rotation=45, fontsize=10)
    elif var == 'selector':
        ax2.semilogx(varlist,df/nf,'b-',label='Primary Files')
        ax2.semilogx(varlist,gf/nf,'b--',label='Ghost Files')
        ax2.semilogx(varlist,cc,'r-',label='Coarse Collisions')
        ax2.semilogx(varlist,rc,'r--',label='Refined Collisions')
    else:
        ax2.plot(varlist,df/nf,'b-',label='Primary Files')
        ax2.plot(varlist,gf/nf,'b--',label='Ghost Files')
        ax2.plot(varlist,cc,'r-',label='Coarse Collisions')
        ax2.plot(varlist,rc,'r--',label='Refined Collisions')
    plt.legend(loc=3,bbox_to_anchor=(0., 1.02, 1., .102),
               ncol=3,mode="expand", borderaxespad=0.)
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(4)
    ax2.tick_params(width=4)
    # Save
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)
        print(fname)

def time_selection(npart_dim, nfiles, fake_regions, 
                   verbose=False, really_verbose=False,
                   decomp='hilbert', order1=6, order2=4, ngz=0,
                   buff=0.5, total_order=10, distrib='uniform',
                   total_regions=True, nreps=10):
    r"""Get performance info for a bitmap selection of a fake dataset.

    Args:
        npart_dim (int): Number of points per dimension in fake dataset.
        nfiles (int): Total number of files the dataset is split accross.
        fake_regions (): Regions that should be selected from the fake dataset
            using a bitmap.
        verbose (bool, optional): If True, information about the bitmap 
            construction is printed. Defaults to False.
        really_verbose (bool, optional): If True, information about the 
            construction of each file in the fake dataset is printed. Defaults 
            to False.
        decomp (str, optional): Name of domain decomposition scheme that should 
            be used to select points for this file. Defaults to 'hilbert'. Valid 
            values include:
              'grid': A perfect grid with no overlap.
              'random': Random points from entire domain.
              'sliced': Points from a slice in one dimension.
              'hilbert': Points from consecutive cells along a Hilbert space
                  filling curve.
              'morton': Points from consecutive cells along a Morton space 
                  filling curve.
            If `decomp.startswith('zoom_')`, then the dataset returned is the 
            will have half the particles distributed across the whole domain and 
            half distributed across one fifth of the domain (at the center), but 
            using scaled domain decompositions. Multiple domain decompositions 
            can be combined by joining them with an underscore. Then the points 
            are split evenly between each scheme.
        order1 (int, optional): Order of the coarse index. Defaults to 6. If 
            None, it is set by `order2` and/or `total_order`.
        order2 (int, optional): Order of the refined index. Defaults to 4. If 
            None, it is set by `order1` and/or `total_order`.
        ngz (int, optional): Number of ghost zones that should be used. Defaults 
            to 0.
        buff (float, optional): Fractional overlap between points in neighboring 
            domain slices. Defaults to 0.5.
        total_order (int, optional): Combined total order of coarse and refined 
            index. Defaults to 10. 
        distrib (str, optional): Description of how points should be distributed 
            across the domain. Defaults to 'uniform'. Valid values include:
              'uniform': Uniform distribution throughout the domain.
              'gaussian': Distribution according to gaussian. Only valid for 
                  `decomp == 'hilbert'`.
        total_regions (bool, optional): If True, times are reported for all 
            regions selected. Otherwise, times are reported individual for each 
            region. Defaults to True.
        nreps (int, optional): Number of times test should be run. Defaults to 
            10.

    """
    # Set order
    if order2 is None:
        if order1 is None:
            order1 = total_order/2
        order2 = total_order - order1
    elif order1 is None:
        order1 = total_order - order2
    # File name
    fname = "bitmap_{}_np{}_nf{}_oc{}_or{}_buff{}".format(decomp,npart_dim,nfiles,
                                                          order1,order2,
                                                          str(buff).replace('.','p'))
    if distrib != 'uniform':
        fname += '_{}'.format(distrib)
    # Fake bitmap
    npart = npart_dim**3
    reg, cc, rc, mem = FakeBitmap(npart, nfiles, order1, order2, decomp=decomp, 
                                  buff=buff, distrib=distrib, fname=fname,
                                  verbose=verbose, really_verbose=really_verbose)
    if total_regions:
        times = np.empty(nreps,dtype='float')
        for k in range(nreps):
            ndf = 0
            ngf = 0
            nf = 0
            times[k] = 0.0
            for fr in fake_regions:
                selector = RegionSelector(fr)
                t1 = time.time()
                df, gf = reg.identify_data_files(selector, ngz=ngz)
                t2 = time.time()
                ndf += len(df)
                ngf += len(gf)
                nf += nfiles
                times[k] += t2-t1
        tt = np.sum(times)
        tm = np.mean(times)
        ts = np.std(times)
    else:
        Nfr = len(fake_regions)
        ndf = np.empty(Nfr, dtype='int32')
        ngf = np.empty(Nfr, dtype='int32')
        nf = np.empty(Nfr, dtype='int32')
        tt = np.empty(Nfr, dtype='float')
        tm = np.empty(Nfr, dtype='float')
        ts = np.empty(Nfr, dtype='float')
        cc = Nfr*[cc]
        rc = Nfr*[rc]
        mem = Nfr*[mem]
        times = np.empty(nreps,dtype='float')
        for i,fr in enumerate(fake_regions):
            selector = RegionSelector(fr)
            for k in range(nreps):
                t1 = time.time()
                df, gf = reg.identify_data_files(selector, ngz=ngz)
                t2 = time.time()
                times[k] = t2-t1
            tt[i] = np.sum(times)
            tm[i] = np.mean(times)
            ts[i] = np.std(times)
            ndf[i] = len(df)
            ngf[i] = len(gf)
            nf[i] = nfiles
    out = dict(tt=tt, tm=tm, ts=ts, ndf=ndf, ngf=ngf, nf=nf, cc=cc, rc=rc, mem=mem)
    return out

def plot_vary_selector(vlist, order1=4, order2=2, 
                       plotfile="vary_selector.png", plot_mem=False, **kws):
    r"""Plot performance of bitmap selection for different selector sizes.

    Args:
        vlist (list of float): Sizes for box shaped selection regions.
        order1 (int, optional): Order of coarse bitmap index. Defaults to 4.
        order2 (int, optional): Ordef of refined bitmap index. Defaults to 2.
        plotfile (str, optional): Name of file where plot should be saved.
            Defaults to 'vary_selector.png'.
        plot_mem (bool, optional): If True, an additional subplot is included 
            showing memory performance.
        **kws: Additional keyword arguments are passed to :meth:`vary_selection_stats`. 

    """
    lw = 5
    mpl.rc('font', weight='bold')#,family='serif')
    mpl.rc('lines',linewidth=lw)
    mpl.rc('axes',linewidth=4)
    # Set up plot
    plt.close('all')
    f, (ax1,ax2) = plt.subplots(2,1,sharex=True)
    ax1.set_yscale('log')
    # Loop over total_order
    for ngz in [0,1]:
        kws['ngz'] = ngz
        kws['order1'] = order1
        kws['order2'] = order2
        kws['plot'] = False
        result = vary_selection_stats('selector', vlist, verbose=True, **kws)
        Nvar = len(vlist)
        t = np.empty(Nvar, dtype='float')
        df = np.empty(Nvar, dtype='float')
        gf = np.empty(Nvar, dtype='float')
        nf = np.empty(Nvar, dtype='float')
        cc = np.empty(Nvar, dtype='float')
        rc = np.empty(Nvar, dtype='float')
        ts = np.empty(Nvar, dtype='float')
        mem = np.empty(Nvar, dtype='float')
        for i,v in enumerate(vlist):
            t[i] = result[v]['tm']
            df[i] = result[v]['ndf']
            gf[i] = result[v]['ngf']
            nf[i] = result[v]['nf']
            cc[i] = float(result[v]['cc'][0])/float(result[v]['cc'][1])
            rc[i] = float(result[v]['rc'][0])/float(result[v]['rc'][1])
            ts[i] = result[v].get('ts',0.0)
            mem[i] = result[v].get('mem',0.0)
        # Plot
        if ngz == 0:
            pfargs = (vlist,df/nf)
            ptkws = dict(linestyle='-',color='k',label='Time w/o Ghost Zones')
            pfkws = dict(linestyle='-',color='k',label='Primary Files')
        else:
            pfargs = (vlist,gf/nf)
            ptkws = dict(linestyle='--',color='b',label='Time w/ Ghost Zones')
            pfkws = dict(linestyle='--',color='b',label='Ghost Files')
            # pckws = dict(linestyle='-.',color='r',label='Collisions')
        if kws.get('plot_errors',False):
            ax1.errorbar(vlist,t,yerr=ts,**ptkws)
        else:
            ax1.plot(vlist,t,**ptkws)
        ax2.plot(*pfargs,**pfkws)
        # ax2.plot(vlist,cc,**pckws)
    # Formatting
    ax1.set_ylabel('Time (s)',fontsize=14, fontweight='bold')
    ax2.set_xlabel('Width of Selector', fontsize=14, fontweight='bold')
    ax2.set_ylabel('% Files Identified', fontsize=14, fontweight='bold')
    ax2.set_ylim((-0.1,1.1))
    for ax in [ax1,ax2]:
        ax.tick_params(width=4)
    plt.legend(loc=3,bbox_to_anchor=(0., 1.02, 1., .102),
               ncol=2,mode="expand", borderaxespad=0.,
               frameon=False)
    # Save
    plt.savefig(plotfile)
    print(plotfile)

def plot_vary_order1(vlist, order2 = 0, 
                     plotfile=None, plot_mem=False, **kws):
    r"""Plot performance of bitmap selection for different index orders.

    Args:
        vlist (list of int): Coarse index orders to test.
        order2 (int, optional): Ordef of refined bitmap index. Defaults to 0.
        plotfile (str, optional): Name of file where plot should be saved.
            Defaults to `'vary_order1_or{}.png'.format(order2)`.
        plot_mem (bool, optional): If True, an additional subplot is included 
            showing memory performance.
        **kws: Additional keyword arguments are passed to :meth:`vary_selection_stats`. 

    """
    lw = 5
    mpl.rc('font', weight='bold')
    mpl.rc('lines',linewidth=lw)
    mpl.rc('axes',linewidth=4)
    if plotfile is None:
        plotfile = 'vary_order1_or{}.png'.format(order2)
    # Set up plot
    plt.close('all')
    if plot_mem:
        f, (ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        f.set_figheight(f.get_figheight()*3.0/2.0)
        ax3.set_yscale('log')
    else:
        f, (ax1,ax2) = plt.subplots(2,1,sharex=True)
    ax1.set_yscale('log')
    # Loop over total_order
    for ngz in [0,1]:
        kws['total_order'] = None
        kws['order2'] = order2
        kws['ngz'] = ngz
        kws['plot'] = False
        kws['extendtag'] = '_or{}'.format(order2)
        result = vary_selection_stats('order1', vlist, verbose=True, **kws)
        Nvar = len(vlist)
        t = np.empty(Nvar, dtype='float')
        df = np.empty(Nvar, dtype='float')
        gf = np.empty(Nvar, dtype='float')
        nf = np.empty(Nvar, dtype='float')
        cc = np.empty(Nvar, dtype='float')
        rc = np.empty(Nvar, dtype='float')
        ts = np.empty(Nvar, dtype='float')
        mem = np.empty(Nvar, dtype='float')
        for i,v in enumerate(vlist):
            t[i] = result[v]['tm']
            df[i] = result[v]['ndf']
            gf[i] = result[v]['ngf']
            nf[i] = result[v]['nf']
            cc[i] = float(result[v]['cc'][0])/float(result[v]['cc'][1])
            rc[i] = float(result[v]['rc'][0])/float(result[v]['rc'][1])
            ts[i] = result[v].get('ts',0.0)
            mem[i] = result[v].get('mem',0.0)/(1024.0*1024.0) # MB
        # Plot
        if ngz == 0:
            pfargs = (vlist,df/nf)
            ptkws = dict(linestyle='-',color='k',label='Time w/o Ghost Zones')
            pfkws = dict(linestyle='-',color='k',label='Primary Files')
        else:
            pfargs = (vlist,gf/nf)
            pcargs = (vlist,cc)
            ptkws = dict(linestyle='--',color='b',label='Time w/ Ghost Zones')
            pfkws = dict(linestyle='--',color='b',label='Ghost Files')
            pckws = dict(linestyle='-.',color='r',label='Collisions')
        if kws.get('plot_errors',False):
            ax1.errorbar(vlist,t,yerr=ts,**ptkws)
        else:
            ax1.plot(vlist,t,**ptkws)
        ax2.plot(*pfargs,**pfkws)
        if ngz == 1:
            ax2.plot(*pcargs,**pckws)
        # Fitting
        if 0:
            def func_mem(x,a,b=2,c=3):#1.81906043):
                return a*(b**(c*x))
            def func_time(x,a,b=1.0,c=1.0):
                return a*(b**(c*(4**x)))
                # return a*(b**(c*0.004*(2**(2*x))))
            def func_pow(x,a,b):
                return a*(b**x)
            print 'fitting memory'
            # print optimization.curve_fit(func_pow, vlist, mem, np.zeros(3))
            fit_mem = optimization.curve_fit(func_mem, vlist, mem, [0.004,2.0,2.0])#,1.0])
            print fit_mem
            ax3.plot(vlist,func_mem(vlist,*fit_mem[0]),'m--')
            fit_mem = optimization.curve_fit(func_pow, vlist, mem, [0.004,4.0])
            print fit_mem
            ax3.plot(vlist,func_pow(vlist,*fit_mem[0]),'c--')
            fmem = func_pow(vlist,*fit_mem[0])
            print 'fitting time'
            fit_time = optimization.curve_fit(func_time, vlist, t, [1.0,1.0])#,1.0])
            print fit_time
            ax1.plot(vlist,func_time(vlist,*fit_time[0]),'m--')
            fit_time = optimization.curve_fit(func_pow, mem, t, [0.003,1.0])
            print fit_time
            ax1.plot(vlist,func_pow(mem,*fit_time[0]),'c--')
    # Formatting
    if plot_mem:
        ax3.plot(vlist,mem,linestyle='-',color='k')
        ax3.set_xlabel('Order of Index', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Size of Index (MB)', fontsize=14, fontweight='bold')
    else:
        ax2.set_xlabel('Order of Index', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (s)',fontsize=14, fontweight='bold')
    ax2.set_ylabel('% Files Identified/\n Cells with Collisions', fontsize=14, fontweight='bold')
    ax2.set_ylim((-0.1,1.1))
    for ax in [ax1,ax2]:
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(4)
        ax.tick_params(width=4)
    plt.sca(ax2)
    plt.legend(loc=3,bbox_to_anchor=(0., 1.02, 1., .102),
               ncol=3,mode="expand", borderaxespad=0.,
               frameon=False)
    # Save
    plt.savefig(plotfile)
    print(plotfile)

def plot_vary_order2(total_order = 6, plotfile=None, plot_mem=False, **kws):
    r"""Plot performance of bitmap selection for different refined index orders.

    Args:
        total_order (int, optional): Total combined order of coarse and refined 
            bitmap indexes. Defaults to 6. This can also be a list of orders. 
            Then a line is plotted for each value.
        plotfile (str, optional): Name of file where plot should be saved.
            Defaults to `'vary_order2_to{}.png'.format(total_order)`
        plot_mem (bool, optional): If True, an additional subplot is included 
            showing memory performance.
        **kws: Additional keyword arguments are passed to :meth:`vary_selection_stats`. 

    """
    lw = 5
    mpl.rc('font', weight='bold')
    mpl.rc('lines',linewidth=lw)
    mpl.rc('axes',linewidth=4)
    if isinstance(total_order, list):
        orders = total_order
    else:
        orders = [total_order]
    # orders = range(2,9)
    if plotfile is None:
        if len(orders)>1:
            plotfile = 'vary_order2_mult.png'
        else:
            plotfile = 'vary_order2_to{}.png'.format(orders[0])
    # Set up plot
    plt.close('all')
    cmap = plt.get_cmap('jet') 
    cnorm = mpl.colors.Normalize(vmin=orders[0], vmax=orders[-1])
    smap = mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap)
    smap.set_array(orders)
    if plot_mem:
        f, (ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        f.set_figheight(f.get_figheight()*3.0/2.0)
        ax3.set_yscale('log')
    else:
        f, (ax1,ax2) = plt.subplots(2,1,sharex=True)
    ax1.set_yscale('log')
    # Loop over total_order
    for o in orders:
        kws['total_order'] = o
        kws['order1'] = None
        kws['plot'] = False
        kws['ngz'] = 0
        kws['extendtag'] = '_to{}'.format(o)
        vlist = range(0,kws['total_order'])
        result = vary_selection_stats('order2',vlist,verbose=True,**kws)
        Nvar = len(vlist)
        t = np.empty(Nvar, dtype='float')
        df = np.empty(Nvar, dtype='float')
        nf = np.empty(Nvar, dtype='float')
        cc = np.empty(Nvar, dtype='float')
        rc = np.empty(Nvar, dtype='float')
        ts = np.empty(Nvar, dtype='float')
        mem = np.empty(Nvar, dtype='float')
        for i,v in enumerate(vlist):
            t[i] = result[v]['tm']
            df[i] = result[v]['ndf']
            nf[i] = result[v]['nf']
            cc[i] = float(result[v]['cc'][0])/float(result[v]['cc'][1])
            rc[i] = float(result[v]['rc'][0])/float(result[v]['rc'][1])
            ts[i] = result[v].get('ts',0.0)
            mem[i] = result[v].get('mem',0.0)/(1024.0*1024.0)
        # Plot
        if len(orders) == 1:
            clr_f = 'k'
            clr_cc = 'b'
            clr_rc = 'r'
        else:
            clr_f = smap.to_rgba(o)
            clr_cc = smap.to_rgba(o)
            clr_rc = smap.to_rgba(o)
        ptkws = dict(linestyle='-',color=clr_f,label='Time')
        pfkws = dict(linestyle='-',color=clr_f)
        pcckws = dict(linestyle='-.',color=clr_cc)
        prckws = dict(linestyle=':',color=clr_rc)
        if o == orders[0]:
            pfkws['label'] = 'Files'
            pcckws['label'] = 'Coarse Coll.'
            prckws['label'] = 'Refined Coll.'
        if kws.get('plot_errors',False):
            ax1.errorbar(vlist,t,yerr=ts,**ptkws)
        else:
            ax1.plot(vlist,t,**ptkws)
        ax2.plot(vlist,df/nf,**pfkws)
        ax2.plot(vlist,cc,**pcckws)
        ax2.plot(vlist,rc,**prckws)
    # Formatting
    if plot_mem:
        ax3.plot(vlist,mem,linestyle='-',color='k')
        ax3.set_xlabel('Order of Refined Index', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Size of Index (MB)', fontsize=14, fontweight='bold')
    else:
        ax2.set_xlabel('Order of Refined Index', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (s)',fontsize=14, fontweight='bold')
    ax2.set_ylabel('% Files Identified/\n Cells with Collisions', fontsize=14, fontweight='bold')
    ax2.set_ylim((-0.1,1.1))
    for ax in [ax1,ax2]:
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(4)
        ax.tick_params(width=4)
    plt.sca(ax2)
    plt.legend(loc=3,bbox_to_anchor=(0., 1.02, 1., .102),
               ncol=3,mode="expand", borderaxespad=0.,
               frameon=False)
    if len(orders) > 1:
        cbar = f.colorbar(smap, ax1,#use_gridspec=True,
                          orientation='horizontal')
        cbar.set_label('Total Order of Combined Indices', 
                       fontsize=14, fontweight='bold')
    # Save
    plt.savefig(plotfile)
    print(plotfile)

def plot_vary_decomp(list_decomp, order1=range(1,8), order2=0, 
                     plotfile=None, plot_collisions=False, plot_mem=False, **kws):
    r"""Plot performance of bitmap selection for different domain decomposition 
    schemes.

    Args:
        vlist (list of str): Domain decomposition schemes.
        order1 (list, optional): Orders of coarse bitmap index to test. Defaults 
            to `range(1,8)`.
        order2 (int, optional): Ordef of refined bitmap index. Defaults to 0.
        plotfile (str, optional): Name of file where plot should be saved.
            Defaults to `'vary_decomp_to{}.png'.format(order2)`.
        plot_mem (bool, optional): If True, an additional subplot is included 
            showing memory performance.
        **kws: Additional keyword arguments are passed to :meth:`vary_selection_stats`. 

    """
    mpl.rc('font', weight='bold')
    mpl.rc('lines',linewidth=5)
    mpl.rc('axes',linewidth=4)
    if plotfile is None:
        plotfile = 'vary_decomp_to{}.png'.format(order2)
    # Set up plot
    plt.close('all')
    cmap = plt.get_cmap('jet') 
    cnorm = mpl.colors.Normalize(vmin=0, vmax=len(list_decomp)-1)
    smap = mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap)
    clrs = ['m','c','r','b']
    stys = [':','-.','--','-']
    if plot_mem:
        f, (ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        f.set_figheight(f.get_figheight()*3.0/2.0)
        ax3.set_yscale('log')
    else:
        f, (ax1,ax2) = plt.subplots(2,1,sharex=True)
    ax1.set_yscale('log')
    # Loop over total_order
    for o,decomp in enumerate(list_decomp):
        kws['total_order'] = None
        kws['decomp'] = decomp
        kws['order2'] = order2
        kws['ngz'] = 0
        kws['plot'] = False
        kws['extendtag'] = '_or{}'.format(order2)
        vlist = order1
        result = vary_selection_stats('order1', vlist, verbose=True, **kws)
        Nvar = len(vlist)
        t = np.empty(Nvar, dtype='float')
        df = np.empty(Nvar, dtype='float')
        gf = np.empty(Nvar, dtype='float')
        nf = np.empty(Nvar, dtype='float')
        cc = np.empty(Nvar, dtype='float')
        mem = np.empty(Nvar, dtype='float')
        # rc = np.empty(Nvar, dtype='float')
        ts = np.empty(Nvar, dtype='float')
        for i,v in enumerate(vlist):
            t[i] = result[v]['tm']
            df[i] = result[v]['ndf']
            gf[i] = result[v]['ngf']
            nf[i] = result[v]['nf']
            cc[i] = float(result[v]['cc'][0])/float(result[v]['cc'][1])
            # rc[i] = float(result[v]['rc'][0])/float(result[v]['rc'][1])
            ts[i] = result[v].get('ts',0.0)
            mem[i] = result[v].get('mem',0.0)
        # Plot
        clr = clrs[o]
        sty = stys[o]
        #clr = smap.to_rgba(o)
        if plot_collisions:
            ptkws = dict(linestyle='-',color=clr,label='{} Time'.format(decomp.title()))
            pfkws = dict(linestyle='-',color=clr,label='{} Files'.format(decomp.title()))
            pckws = dict(linestyle='--',color=clr,label='{} Collisions'.format(decomp.title()))
            pmkws = dict(linestyle='-',color=clr,label='{} Memory'.format(decomp.title()))
        else:
            ptkws = dict(linestyle=sty,color=clr,label=decomp.title())
            pfkws = dict(linestyle=sty,color=clr,label=decomp.title())
            pmkws = dict(linestyle=sty,color=clr,label=decomp.title())
        if kws.get('plot_errors',False):
            ax1.errorbar(vlist,t,yerr=ts,**ptkws)
        else:
            ax1.plot(vlist,t,**ptkws)
        ax2.plot(vlist,df/nf,**pfkws)
        if plot_collisions:
            ax2.plot(vlist,cc,**pckws)
        if plot_mem:
            ax3.plot(vlist,mem,**pmkws)
    # Formatting
    if plot_mem:
        ax3.set_xlabel('Order of Index', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Size of Index (MB)', fontsize=14, fontweight='bold')
    else:
        ax2.set_xlabel('Order of Index', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (s)',fontsize=14, fontweight='bold')
    if plot_collisions:
        ax2.set_ylabel('% files/collisions', fontsize=14, fontweight='bold')
    else:
        ax2.set_ylabel('% Files Identified', fontsize=14, fontweight='bold')
    ax2.set_ylim((-0.1,1.1))
    for ax in [ax1,ax2]:
        ax.tick_params(width=4)
    plt.sca(ax2)
    plt.legend(loc=9,bbox_to_anchor=(0., 2.07, 1., .102),
               ncol=2,mode="expand", borderaxespad=0.,
               frameon=False)
    # Save
    plt.savefig(plotfile)
    print(plotfile)

def make_all():
    r"""Create all plots necessary for the paper."""
    kws = dict(
        nfiles = 512,          # Number of files fake dataset should be split accross
        npart_dim = 1024,      # Number of particles per dimension in fake dataset
        DLE = [0.0, 0.0, 0.0], # Left edges of dataset domain
        DRE = [1.0, 1.0, 1.0], # Right edges of dataset domain
        decomp = 'hilbert',    # Domain decomposition used to split data between files
        buff = 0.1,            # Fraction of scatter there should be between addjacent 
                               # cells in the domain decomposition
        distrib = 'uniform',   # Distribution of particles within the domain
        ngz = 0,               # Number of ghost zones that should be included
        nreps = 10)            # Number of times test should be repeated
    # Figure 7: Performance vs. Total Index Order
    list_order1 = np.arange(1,8)
    plot_vary_order1(list_order1, order2 = 0,
                     plotfile="figure7.png", plot_mem=True, **kws)
    # Figure 8: Performance vs. Refined Index Order
    plot_vary_order2(total_order = 6,
                     plotfile="figure8.png", plot_mem=False, **kws)
    # Figure 9: Performance vs. Selector Size
    sizes = np.logspace(-1, 0, num=20, endpoint=True)
    plot_vary_selector(sizes, order1 = 4, order2 = 2,
                       plotfile="figure9.png", plot_mem=False, **kws)
    # Figure 10: Performance vs. Domain Decomposition
    list_decomp = ['random','sliced','morton','hilbert']
    plot_vary_decomp(list_decomp, order1 = range(1,8), order2=0,
                     plotfile="figure10.png", plot_mem=True,
                     plot_collisions=False, **kws)

if __name__=="__main__":
    make_all()

