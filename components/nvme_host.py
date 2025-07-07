import simpy
import random
from base.traffic_generator import TrafficGenerator
from base.packet import Packet
from components.nvme_driver import NVMe_Driver # New import

class NVMeHost:
    """
    A simplified NVMe Host model, composed of a TrafficGenerator and an NVMe_Driver.
    It orchestrates command generation and delegates command sending/completion processing to the driver.
    """
    def __init__(self, env, req_port, resp_port, nvme_slave_idx):
        self.env = env
        self.req_port = req_port # To SoC Bus
        self.resp_port = resp_port # From SoC Bus
        self.nvme_slave_idx = nvme_slave_idx
        
        self.next_cmd_id = 0
        
        # Internal queue to connect TrafficGenerator to NVMe_Driver
        self.traffic_gen_out_queue = simpy.Store(env)

        # Define a function to generate NVMe command Packets
        def nvme_command_generator():
            cmd_id = self.next_cmd_id
            self.next_cmd_id += 1
            
            cmd_type = random.choice(['read', 'write', 'identify'])
            lba = random.randint(0, 250) # Assuming 256 blocks total
            num_blocks = random.randint(1, 4)
            data_payload = b'\xCC' * (num_blocks * 4096) # Assuming 4096 bytes/block

            return Packet(
                id=cmd_id,
                type=cmd_type,
                source_id=f"NVMeHost_{self.nvme_slave_idx}",
                destination_id=f"NVMeSlave_{self.nvme_slave_idx}",
                address=lba,
                size=num_blocks,
                data=data_payload,
                nvme_cmd_type=cmd_type,
                nvme_lba=lba,
                nvme_num_blocks=num_blocks,
                nvme_data_payload=data_payload
            )

        # Instantiate TrafficGenerator
        self.traffic_gen = TrafficGenerator(
            env=env,
            out_port=self.traffic_gen_out_queue, # Connects to driver's input
            interval=random.randint(50, 200),
            item_generator_func=nvme_command_generator,
            num_items=None
        )

        # Instantiate NVMe_Driver
        self.nvme_driver = NVMe_Driver(
            env=env,
            cmd_in_port=self.traffic_gen_out_queue, # Receives commands from TrafficGenerator
            req_out_port=self.req_port, # Sends requests to SoC Bus
            resp_in_port=self.resp_port, # Receives responses from SoC Bus
            nvme_slave_idx=self.nvme_slave_idx
        )
        
        print(f"[{self.env.now}] NVMeHost: Initialized. TrafficGen and Driver are active.")