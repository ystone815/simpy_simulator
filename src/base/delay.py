import simpy
from .packet import Packet # New import

class DelayLine:
    def __init__(self, env, in_port, out_port, delay):
        self.env = env
        self.in_port = in_port
        self.out_port = out_port
        self.delay = delay
        self.action = env.process(self.run())

    def run(self):
        while True:
            packet = yield self.in_port.get() # Expecting a Packet object
            yield self.env.timeout(self.delay)
            yield self.out_port.put(packet) # Pass the Packet object

class DelayLineUtil:
    def __init__(self, env, in_port, out_port, data_path_width_byte, clock_freq_mhz, utilization=1.0):
        if not (0 < utilization <= 1.0):
            raise ValueError("Utilization must be between 0 and 1.")
        if not (data_path_width_byte > 0 and clock_freq_mhz > 0):
            raise ValueError("Data path width and clock frequency must be positive.")
            
        self.env = env
        self.in_port = in_port
        self.out_port = out_port
        self.data_path_width_byte = data_path_width_byte
        self.clock_freq_mhz = clock_freq_mhz
        self.utilization = utilization
        
        self.clock_period_ns = 1000 / self.clock_freq_mhz
        self.resource = simpy.Resource(env, capacity=1)
        self.last_event_time = 0
        self.busy_time = 0
        self.action = env.process(self.run())

    def run(self):
        while True:
            packet = yield self.in_port.get() # Expecting a Packet object
            
            # Use packet.size for delay calculation
            item_size = packet.size if packet.size is not None else 1 # Default to 1 if size is not set
            delay = self._get_delay_ns(item_size)
            
            with self.resource.request() as req:
                start_wait = self.env.now
                yield req
                end_wait = self.env.now
                
                self.busy_time += (end_wait - self.last_event_time) * (1 if self.resource.count > 0 else 0)
                
                yield self.env.timeout(delay)
                yield self.out_port.put(packet) # Pass the Packet object
                
                self.last_event_time = self.env.now

    def _get_delay_ns(self, item_size_in_bytes):
        if item_size_in_bytes <= 0:
            return 0
        
        num_cycles = -(-item_size_in_bytes // self.data_path_width_byte)
        ideal_delay = num_cycles * self.clock_period_ns
        actual_delay = ideal_delay / self.utilization
        
        return actual_delay

    def get_utilization(self):
        if self.env.now == 0:
            return 0
        return (self.busy_time / self.env.now) * 100