
import simpy

class Mux:
    def __init__(self, env, in_ports, out_port):
        self.env = env
        self.in_ports = in_ports
        self.out_port = out_port
        self.env.process(self.run())

    def run(self):
        # This Mux merges multiple input ports into one output port.
        # It uses simpy.AnyOf to wait for an item on any of the input ports.
        while True:
            # Create a list of get events from all input ports
            get_events = [port.get() for port in self.in_ports]
            # Wait for the first event to be triggered
            result = yield self.env.any_of(get_events)
            # The result is a dictionary {event: value}, get the first value
            msg = next(iter(result.values()))
            yield self.out_port.put(msg)

class Demux:
    def __init__(self, env, in_port, out_ports):
        self.env = env
        self.in_port = in_port
        self.out_ports = out_ports
        self.env.process(self.run())

    def run(self):
        while True:
            msg = yield self.in_port.get()
            # The message must contain routing info: (destination_index, data)
            try:
                dest, data = msg
                if 0 <= dest < len(self.out_ports):
                    yield self.out_ports[dest].put(data)
                else:
                    print(f"[{self.env.now}] Demux Error: Invalid destination index {dest}")
            except (ValueError, TypeError):
                print(f"[{self.env.now}] Demux Error: Invalid message format. Expected (dest, data), got {msg}")
