import pycuda.autoinit
import pycuda.driver as drv
import numpy
import signal
import colorama
from colorama import Fore
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void multiply_them(float *dest, float *res_1, float *res_2)
{
  const int i = threadIdx.x;
  dest[i] = res_1[i] * res_2[i];
}
""")

def ctrlc(sig, frame):
    raise KeyboardInterrupt(Fore.BLUE + "CTRL-C!")

print("CUDA Stress Test Started")
while True:
    multiply_them = mod.get_function("multiply_them")

    a = numpy.random.randn(999).astype(numpy.float32)
    b = numpy.random.randn(999).astype(numpy.float32)
    c = numpy.random.randn(999).astype(numpy.float32)
    d = numpy.random.randn(999).astype(numpy.float32)

    res_1 = a * b
    res_2 = c * d

    signal.signal(signal.SIGINT, ctrlc)

    dest = numpy.zeros_like(a)
    multiply_them(
        drv.Out(dest), drv.In(res_1), drv.In(res_2),
        block=(999, 1, 1), grid=(1, 1))


print(dest-res_1*res_2)
