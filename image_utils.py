import numpy as np
import scipy
import scipy.sparse as sps
import scipy.ndimage as spi
import scipy.misc as spm
from numpy.lib.stride_tricks import as_strided


def rolling_block(A, block=(3, 3)):
  shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
  strides = (A.strides[0], A.strides[1]) + A.strides
  return as_strided(A, shape=shape, strides=strides)


def compute_matting_laplacian(img, eps=10 ** (-7), win_rad=1):
  win_size = (win_rad * 2 + 1) ** 2
  h, w, d = img.shape
  # Number of window centre indices in h, w axes
  # c_h, c_w = h - win_rad - 1, w - win_rad - 1
  c_h, c_w = h - win_rad * 2, w - win_rad * 2
  win_diam = win_rad * 2 + 1

  indsM = np.arange(h * w).reshape((h, w))
  ravelImg = img.reshape(h * w, d)
  win_inds = rolling_block(indsM, block=(win_diam, win_diam))

  win_inds = win_inds.reshape(c_h, c_w, win_size)
  winI = ravelImg[win_inds]

  win_mu = np.mean(winI, axis=2, keepdims=True)
  win_var = np.einsum('...ji,...jk ->...ik', winI, winI) / win_size - np.einsum('...ji,...jk ->...ik', win_mu, win_mu)

  inv = np.linalg.inv(win_var + (eps / win_size) * np.eye(3))
  X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv)
  vals = np.eye(win_size) - (1 / win_size) * (1 + np.einsum('...ij,...kj->...ik', X, winI - win_mu))

  nz_indsCol = np.tile(win_inds, win_size).ravel()
  nz_indsRow = np.repeat(win_inds, win_size).ravel()
  nz_indsVal = vals.ravel()
  L = scipy.sparse.coo_matrix((nz_indsVal, (nz_indsRow, nz_indsCol)), shape=(h * w, h * w)).astype(np.float32)
  return L


def getlaplacian(i_arr, consts, epsilon=0.0000001, win_size=1):
  neb_size = (win_size * 2 + 1) ** 2
  h, w, c = i_arr.shape
  img_size = w * h
  consts = spi.morphology.grey_erosion(consts, footprint=np.ones(shape=(win_size * 2 + 1, win_size * 2 + 1)))

  indsM = np.reshape(np.array(range(img_size)), newshape=(h, w), order='F')
  tlen = int((-consts[win_size:-win_size, win_size:-win_size] + 1).sum() * (neb_size ** 2))

  row_inds = np.zeros(tlen)
  col_inds = np.zeros(tlen)
  vals = np.zeros(tlen)
  l = 0
  for j in range(win_size, w - win_size):
    for i in range(win_size, h - win_size):
      if consts[i, j]:
        continue
      win_inds = indsM[i - win_size:i + win_size + 1, j - win_size: j + win_size + 1]
      win_inds = win_inds.ravel(order='F')
      win_i = i_arr[i - win_size:i + win_size + 1, j - win_size: j + win_size + 1, :]
      win_i = win_i.reshape((neb_size, c), order='F')
      win_mu = np.mean(win_i, axis=0).reshape(1, win_size * 2 + 1)
      win_var = np.linalg.inv(np.matmul(win_i.T, win_i) / neb_size - np.matmul(win_mu.T, win_mu) + epsilon / neb_size * np.identity(c))

      win_i2 = win_i - win_mu
      tvals = (1 + np.matmul(np.matmul(win_i2, win_var), win_i2.T)) / neb_size

      ind_mat = np.broadcast_to(win_inds, (neb_size, neb_size))
      row_inds[l: (neb_size ** 2 + l)] = ind_mat.ravel(order='C')
      col_inds[l: neb_size ** 2 + l] = ind_mat.ravel(order='F')
      vals[l: neb_size ** 2 + l] = tvals.ravel(order='F')
      l += neb_size ** 2

  vals = vals.ravel(order='F')
  row_inds = row_inds.ravel(order='F')
  col_inds = col_inds.ravel(order='F')
  a_sparse = sps.csr_matrix((vals, (row_inds, col_inds)), shape=(img_size, img_size))

  sum_a = a_sparse.sum(axis=1).T.tolist()[0]
  a_sparse = sps.diags([sum_a], [0], shape=(img_size, img_size)) - a_sparse

  return a_sparse.tocoo().astype(np.float32)