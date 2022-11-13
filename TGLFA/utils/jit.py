from torch.utils.cpp_extension import load

smo = load("smo", ["TGLFA/utils/smo_cuda.cpp", "TGLFA/utils/smo_cuda_kernel.cu"], verbose=True)