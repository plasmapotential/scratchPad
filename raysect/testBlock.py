from raysect.core import MulticoreEngine
from raysect.optical.observer import PinholeCamera
import time
camera = PinholeCamera((512, 512))
t0 = time.time()
# allowing the camera to use all available CPU cores.
camera.render_engine = MulticoreEngine()
t1 = time.time()
# or forcing the render engine to use a specific number of CPU processes
camera.render_engine = MulticoreEngine(processes=1)
t2 = time.time()

print("Multicore: {:f}".format(t1-t0))
print("Serial: {:f}".format(t2-t1))
