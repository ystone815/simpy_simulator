
import simpy
import random
from .packet import Packet # New import

class TrafficGenerator:
    """
    A general-purpose traffic generator that periodically creates Packet objects
    and puts them onto an output port.
    """
    def __init__(self, env, out_port, interval, item_generator_func, num_items=None):
        self.env = env
        self.out_port = out_port
        self.interval = interval
        self.item_generator_func = item_generator_func
        self.num_items = num_items
        self.generated_count = 0

        self.action = env.process(self.run())
        print(f"[{self.env.now}] TrafficGenerator: Initialized. Interval: {self.interval}, Num Items: {self.num_items if self.num_items is not None else 'Infinite'}")

    def run(self):
        while True:
            if self.num_items is not None and self.generated_count >= self.num_items:
                print(f"[{self.env.now}] TrafficGenerator: Finished generating {self.generated_count} items.")
                break
            
            # Generate item (expected to be a Packet object)
            packet = self.item_generator_func()
            packet.timestamp = self.env.now # Set the generation timestamp
            
            # Put packet onto the output port
            print(f"[{self.env.now}] TrafficGenerator: Generated {packet.type} packet (ID: {packet.id}).")
            yield self.out_port.put(packet)
            self.generated_count += 1
            
            # Wait for the next interval
            yield self.env.timeout(self.interval)
