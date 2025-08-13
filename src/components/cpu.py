import simpy
from src.base.packet import Packet # New import

class Cpu:
    def __init__(self, env, req_port, resp_port, cpu_id=0):
        self.env = env
        self.req_port = req_port
        self.resp_port = resp_port
        self.cpu_id = cpu_id
        self.next_transaction_id = 0
        self.action = env.process(self.run())

    def run(self):
        print(f"[{self.env.now}] CPU {self.cpu_id}: Starting up.")
        
        # Example: Write to memory (target Memory is Slave 0 on SoC Bus)
        mem_slave_idx = 0
        addr = 0
        data = b'Hello SimPy!'
        
        write_packet = Packet(
            id=self.next_transaction_id,
            type='write',
            source_id=f"CPU_{self.cpu_id}",
            destination_id=f"Memory",
            address=addr,
            size=len(data),
            data=data,
            timestamp=self.env.now
        )
        self.next_transaction_id += 1

        print(f"[{self.env.now}] CPU {self.cpu_id}: Requesting write to Memory at addr {addr} with data '{data.decode()}' (ID: {write_packet.id}).")
        yield self.req_port.put((mem_slave_idx, write_packet)) # (target_slave_idx, Packet)
        
        response_packet = yield self.resp_port.get() # Expecting a Packet object
        print(f"[{self.env.now}] CPU {self.cpu_id}: Write response for ID {response_packet.id}: Status={response_packet.status}, Error={response_packet.error_message}.")
        
        # Example: Read from memory
        read_size = len(data)
        read_packet = Packet(
            id=self.next_transaction_id,
            type='read',
            source_id=f"CPU_{self.cpu_id}",
            destination_id=f"Memory",
            address=addr,
            size=read_size,
            timestamp=self.env.now
        )
        self.next_transaction_id += 1

        print(f"[{self.env.now}] CPU {self.cpu_id}: Requesting read from Memory at addr {addr} with size {read_size} (ID: {read_packet.id}).")
        yield self.req_port.put((mem_slave_idx, read_packet))
        
        response_packet = yield self.resp_port.get()
        print(f"[{self.env.now}] CPU {self.cpu_id}: Read response for ID {response_packet.id}: Status={response_packet.status}, Data={response_packet.data.decode() if response_packet.data else None}, Error={response_packet.error_message}.")

        print(f"[{self.env.now}] CPU {self.cpu_id}: Idling...")
        while True:
            yield self.env.timeout(100) # CPU is just idling after initial tasks