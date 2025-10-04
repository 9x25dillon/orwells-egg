import os


JULIA_BASE = os.getenv("JULIA_BASE", "http://localhost:9000")
# baseline visco parameters
BASE_CAPACITY = int(os.getenv("BASE_CAPACITY", 8))
TAU_SECONDS = float(os.getenv("TAU_SECONDS", 4.0))
