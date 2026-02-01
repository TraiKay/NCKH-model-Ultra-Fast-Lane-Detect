from setuptools import setup
import os
print("DEBUG: Setting env vars")
os.environ["CUDA_HOME"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
os.environ["PATH"] = os.environ["CUDA_HOME"] + r"\bin;" + os.environ["PATH"]
print(f"DEBUG: CUDA_HOME={os.environ.get('CUDA_HOME')}")

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_interp',
    ext_modules=[
        CUDAExtension('my_interp', [
            'my_interp_cuda.cpp',
            'my_interp_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
