
import simpy

class Memory:
    def __init__(self, env, size_in_bytes, in_port, out_port, access_latency_ns=1):
        if not (size_in_bytes > 0 and access_latency_ns >= 0):
            raise ValueError("Memory size must be positive and latency must be non-negative.")
            
        self.env = env
        self.size = size_in_bytes
        self.access_latency_ns = access_latency_ns
        self.data = bytearray(size_in_bytes)
        
        self.in_port = in_port
        self.out_port = out_port
        
        self.action = env.process(self.run())

    def run(self):
        while True:
            request = yield self.in_port.get()
            yield self.env.timeout(self.access_latency_ns)
            
            try:
                op, addr, *args = request
                
                if op == 'write':
                    data = args[0]
                    if not isinstance(data, (bytes, bytearray)):
                        raise TypeError("Write data must be bytes or bytearray")
                    
                    if addr + len(data) > self.size:
                        raise IndexError("Write address out of bounds")
                        
                    self.data[addr:addr + len(data)] = data
                    yield self.out_port.put(('OK',))
                    
                elif op == 'read':
                    size = args[0]
                    if not isinstance(size, int) or size <= 0:
                        raise TypeError("Read size must be a positive integer")

                    if addr + size > self.size:
                        raise IndexError("Read address out of bounds")
                        
                    read_data = self.data[addr:addr + size]
                    yield self.out_port.put(('OK', read_data))
                    
                else:
                    raise ValueError(f"Unknown operation: {op}")

            except (ValueError, IndexError, TypeError) as e:
                print(f"[{self.env.now}] Memory Error: {e}")
                yield self.out_port.put(('ERROR', str(e)))
