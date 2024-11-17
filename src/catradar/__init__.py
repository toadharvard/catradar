import os

import taichi as ti

if os.environ.get("GITHUB_ACTIONS") == "true":
    print("Testing mode, running on the cpu")
    ti.init(arch=ti.cpu, debug=True)
else:
    print("Standard mode, running on the gpu")
    ti.init(arch=ti.gpu)
