
import simpy

class NVMeSlaveDevice:
    """
    A simplified NVMe Slave Device model.
    It receives commands via in_port (Submission Queue) and sends completions via out_port (Completion Queue).
    It simulates basic Read/Write operations on an internal memory.
    """
    def __init__(self, env, in_port, out_port, size_in_blocks=1024, block_size_bytes=4096, processing_latency_ns=100):
        self.env = env
        self.in_port = in_port
        self.out_port = out_port
        self.size_in_blocks = size_in_blocks
        self.block_size_bytes = block_size_bytes
        self.processing_latency_ns = processing_latency_ns
        
        self.total_size_bytes = size_in_blocks * block_size_bytes
        self.memory = bytearray(self.total_size_bytes) # Internal simulated NAND/DRAM
        
        self.action = env.process(self.run())
        print(f"[{self.env.now}] NVMeSlaveDevice: Initialized with {self.size_in_blocks} blocks ({self.total_size_bytes / (1024*1024):.2f} MB).")

    def run(self):
        while True:
            # Commands are expected to come from the SoCBus, which prepends master_idx
            master_idx, command = yield self.in_port.get()
            
            cmd_id, cmd_type, lba, num_blocks, data_payload = self._parse_command(command)
            
            print(f"[{self.env.now}] NVMeSlaveDevice: Received command {cmd_type} (ID: {cmd_id}) from Master {master_idx} for LBA {lba}, {num_blocks} blocks.")
            
            # Simulate processing delay
            yield self.env.timeout(self.processing_latency_ns)
            
            status = 'SUCCESS'
            bytes_transferred = 0
            read_data_payload = None
            
            try:
                if cmd_type == 'write':
                    start_byte = lba * self.block_size_bytes
                    end_byte = start_byte + (num_blocks * self.block_size_bytes)
                    
                    if end_byte > self.total_size_bytes or start_byte < 0:
                        raise IndexError("Write address out of bounds")
                    if len(data_payload) != (num_blocks * self.block_size_bytes):
                        raise ValueError("Data payload size mismatch for write command")
                        
                    self.memory[start_byte:end_byte] = data_payload
                    bytes_transferred = len(data_payload)
                    print(f"[{self.env.now}] NVMeSlaveDevice: Wrote {bytes_transferred} bytes to LBA {lba}.")
                    
                elif cmd_type == 'read':
                    start_byte = lba * self.block_size_bytes
                    end_byte = start_byte + (num_blocks * self.block_size_bytes)
                    
                    if end_byte > self.total_size_bytes or start_byte < 0:
                        raise IndexError("Read address out of bounds")
                        
                    read_data_payload = self.memory[start_byte:end_byte]
                    bytes_transferred = len(read_data_payload)
                    print(f"[{self.env.now}] NVMeSlaveDevice: Read {bytes_transferred} bytes from LBA {lba}.")
                    
                elif cmd_type == 'identify':
                    # Simplified identify data
                    read_data_payload = b'NVMe_Controller_ID_Data'
                    bytes_transferred = len(read_data_payload)
                    print(f"[{self.env.now}] NVMeSlaveDevice: Processed Identify command.")

                else:
                    status = 'ERROR'
                    print(f"[{self.env.now}] NVMeSlaveDevice: Unsupported command type: {cmd_type}")

            except (IndexError, ValueError, TypeError) as e:
                status = 'ERROR'
                print(f"[{self.env.now}] NVMeSlaveDevice Error: {e}")
            
            # Create completion entry
            completion = (cmd_id, status, bytes_transferred, read_data_payload)
            
            # Send completion back to the master via the bus
            yield self.out_port.put((master_idx, completion))
            print(f"[{self.env.now}] NVMeSlaveDevice: Sent completion for command ID {cmd_id} with status {status}.")

    def _parse_command(self, command):
        # Helper to parse the command tuple with default values
        cmd_id = command[0] if len(command) > 0 else None
        cmd_type = command[1] if len(command) > 1 else None
        lba = command[2] if len(command) > 2 else 0
        num_blocks = command[3] if len(command) > 3 else 0
        data_payload = command[4] if len(command) > 4 else b''
        return cmd_id, cmd_type, lba, num_blocks, data_payload
