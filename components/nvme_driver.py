
import simpy
from base.packet import Packet

class NVMe_Driver:
    """
    A dedicated NVMe Driver class that handles sending commands to the NVMe Slave
    and processing completions. It receives commands as Packet objects.
    """
    def __init__(self, env, cmd_in_port, req_out_port, resp_in_port, nvme_slave_idx):
        self.env = env
        self.cmd_in_port = cmd_in_port # From TrafficGenerator (via NVMeHost)
        self.req_out_port = req_out_port # To SoC Bus
        self.resp_in_port = resp_in_port # From SoC Bus
        self.nvme_slave_idx = nvme_slave_idx
        
        self.outstanding_commands = {} # To track commands and their Packet objects

        self.action = env.process(self.run())
        print(f"[{self.env.now}] NVMe_Driver: Initialized. Target Slave Index: {self.nvme_slave_idx}")

    def run(self):
        # Start the command processing and completion handling processes
        self.env.process(self._process_commands())
        self.env.process(self._completion_handler())
        
        while True:
            yield self.env.timeout(1000) # Keep the driver process alive

    def _process_commands(self):
        """Takes command packets from the input queue and sends them to the bus."""
        while True:
            packet = yield self.cmd_in_port.get() # Get Packet object from TrafficGenerator
            
            self.outstanding_commands[packet.id] = packet # Store the Packet object for completion tracking
            
            # Send command to the SoC Bus, specifying the target NVMe Slave index
            print(f"[{self.env.now}] NVMe_Driver: Sending {packet.type} command (ID: {packet.id}) to Slave {self.nvme_slave_idx}.")
            yield self.req_out_port.put((self.nvme_slave_idx, packet)) # Pass the Packet object directly

    def _completion_handler(self):
        """Receives completion packets from the bus and processes them."""
        while True:
            # Completions come from the bus, which already routed them to this driver's resp_in_port
            completion_packet = yield self.resp_in_port.get()
            
            cmd_id = completion_packet.id
            status = completion_packet.status
            
            if cmd_id in self.outstanding_commands:
                original_packet = self.outstanding_commands.pop(cmd_id)
                latency = self.env.now - original_packet.timestamp
                print(f"[{self.env.now}] NVMe_Driver: Command ID {cmd_id} ({original_packet.type}) completed with status {status}. Latency: {latency:.2f} ns.")
                if status == 'ERROR':
                    print(f"[{self.env.now}] NVMe_Driver: Error details: {completion_packet.error_message}")
            else:
                print(f"[{self.env.now}] NVMe_Driver: Received unexpected completion for ID {cmd_id}.")
