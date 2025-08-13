
import simpy

class SramSlave:
    """
    A simple SRAM slave model that uses an external port-based connection.

    It receives requests via 'in_port' and sends responses via 'out_port'.
    It models a fixed access latency for both read and write operations.
    """
    def __init__(self, env, size_in_bytes, in_port, out_port, access_latency_ns=10):
        """
        - env: The simpy simulation environment.
        - size_in_bytes: The total capacity of the SRAM.
        - in_port: The simpy.Store object for incoming requests.
        - out_port: The simpy.Store object for outgoing responses.
        - access_latency_ns: The time it takes for one access (read or write).
        """
        if not (size_in_bytes > 0 and access_latency_ns >= 0):
            raise ValueError("SRAM size must be positive and latency must be non-negative.")
            
        self.env = env
        self.size = size_in_bytes
        self.access_latency_ns = access_latency_ns
        self.memory = bytearray(size_in_bytes)
        
        # Connect to external ports
        self.in_port = in_port
        self.out_port = out_port
        
        # Start the SRAM's main process
        self.action = env.process(self.run())

    def run(self):
        """The main process loop for the SRAM slave."""
        while True:
            # 1. Wait for a request from a master via the input port
            request = yield self.in_port.get()
            
            # 2. Simulate access latency
            yield self.env.timeout(self.access_latency_ns)
            
            # 3. Process the request
            try:
                op, addr, *args = request
                
                if op == 'write':
                    data = args[0]
                    if not isinstance(data, (bytes, bytearray)):
                        raise TypeError("Write data must be bytes or bytearray")
                    
                    if addr + len(data) > self.size:
                        raise IndexError("Write address out of bounds")
                        
                    # Perform the write
                    self.memory[addr:addr + len(data)] = data
                    # Send response via the output port
                    yield self.out_port.put(('OK',))
                    
                elif op == 'read':
                    size = args[0]
                    if not isinstance(size, int) or size <= 0:
                        raise TypeError("Read size must be a positive integer")

                    if addr + size > self.size:
                        raise IndexError("Read address out of bounds")
                        
                    # Perform the read
                    read_data = self.memory[addr:addr + size]
                    # Send response via the output port
                    yield self.out_port.put(('OK', read_data))
                    
                else:
                    raise ValueError(f"Unknown operation: {op}")

            except (ValueError, IndexError, TypeError) as e:
                # Handle errors by sending an error response
                print(f"SRAM Error at {self.env.now}: {e}")
                yield self.out_port.put(('ERROR', str(e)))

