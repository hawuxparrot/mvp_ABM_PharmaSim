import pickle
from runtime.native_bridge import create_native_simulator

compiled = pickle.load(open("/tmp/compiled.pkl", "rb"))
sim = create_native_simulator(compiled)
sim.run_ticks(20_000)