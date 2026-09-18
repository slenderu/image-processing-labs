# Camera shake blur kernels

These eight kernels were measured from real camera shake and published with

> A. Levin, Y. Weiss, F. Durand, and W. T. Freeman, "Understanding and
> Evaluating Blind Deconvolution Algorithms," IEEE Conference on Computer
> Vision and Pattern Recognition, 2009.

They are the standard benchmark set for image deblurring.  The authors
distribute them inside a MATLAB dataset at

    https://webee.technion.ac.il/people/anat.levin/papers/LevinEtalCVPR09Data.rar

and the paper is at

    https://webee.technion.ac.il/people/anat.levin/papers/deconvLevinEtalCVPR09.pdf

The files here hold the same numbers, written as plain text so that no
extra software is needed to read them.  They are reproduced for use in this
course, with attribution to the authors above.  Please cite the paper if
you use them.

## The files

| File                  | Size    | Nonzero entries |
| --------------------- | ------- | --------------- |
| `levin09_kernel1.txt` | 19 x 19 | 109 |
| `levin09_kernel2.txt` | 17 x 17 | 114 |
| `levin09_kernel3.txt` | 15 x 15 |  87 |
| `levin09_kernel4.txt` | 27 x 27 | 200 |
| `levin09_kernel5.txt` | 13 x 13 |  94 |
| `levin09_kernel6.txt` | 21 x 21 | 136 |
| `levin09_kernel7.txt` | 23 x 23 | 175 |
| `levin09_kernel8.txt` | 23 x 23 | 166 |

Every kernel is nonnegative, sums to exactly 1 in double precision, and has
an odd size in both axes.  Because the entries are nonnegative and sum to
1, the largest singular value of the convolution operator is at most 1.

Each file has one row of the kernel per line, with the values separated by
spaces, written to full double precision.  Reading one back gives exactly
the numbers in the original MATLAB file.

## Reading a kernel

```python
import numpy as np
import torch

k = torch.tensor(np.loadtxt("levin09_kernel1.txt"), dtype=torch.float64)
print(k.shape, k.sum())        # torch.Size([19, 19]) tensor(1., dtype=torch.float64)
```

## How these files were made

The archive above was unpacked, and the variable `f` was read from
`im05_flit01.mat` through `im05_flit08.mat` with `scipy.io.loadmat`.  The
same eight kernels appear in the files for all four test images, so the
choice of image does not matter.  Each array was written with
`numpy.savetxt` using the format `%.17g`, which round trips exactly.
