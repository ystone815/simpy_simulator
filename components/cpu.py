
import simpy

class Cpu:
    def __init__(self, env, req_port, resp_port):
        self.env = env
        self.req_port = req_port
        self.resp_port = resp_port
        self.action = env.process(self.run())

    def run(self):
        print(f"[{self.env.now}] CPU: Starting up.")
        
        # Example: Write to memory
        addr = 0
        data = b'Hello SimPy!'
        print(f"[{self.env.now}] CPU: Requesting write to addr {addr} with data '{data.decode()}'")
        yield self.req_port.put(('write', addr, data))
        response = yield self.resp_port.get()
        print(f"[{self.env.now}] CPU: Write response: {response}")
        
        # Example: Read from memory
        read_size = len(data)
        print(f"[{self.env.now}] CPU: Requesting read from addr {addr} with size {read_size}")
        yield self.req_port.put(('read', addr, read_size))
        response = yield self.resp_port.get()
        print(f"[{self.env.now}] CPU: Read response: {response}")

        print(f"[{self.env.now}] CPU: Idling...")
        while True:
            yield self.env.timeout(100) # CPU is just idling after initial tasks
