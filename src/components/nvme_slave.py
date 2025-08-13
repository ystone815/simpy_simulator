import simpy
from src.base.packet import Packet
from src.components.sram import SramSlave # New import

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
        
        # Internal SRAM channels
        self.sram_req_channel = simpy.Store(env)
        self.sram_resp_channel = simpy.Store(env)

        # Instantiate internal SRAM Slave
        self.internal_sram = SramSlave(
            env=env,
            size_in_bytes=self.total_size_bytes,
            in_port=self.sram_req_channel,
            out_port=self.sram_resp_channel,
            access_latency_ns=1 # Assuming fast internal SRAM access
        )
        
        self.action = env.process(self.run())
        print(f"[{self.env.now}] NVMeSlaveDevice: Initialized with {self.size_in_blocks} blocks ({self.total_size_bytes / (1024*1024):.2f} MB) using internal SRAM.")

    def run(self):
        while True:
            master_idx, request_packet = yield self.in_port.get()
            
            cmd_id = request_packet.id
            cmd_type = request_packet.type
            lba = request_packet.address
            num_blocks = request_packet.size
            data_payload = request_packet.data
            
            print(f"[{self.env.now}] NVMeSlaveDevice: Received {cmd_type} command (ID: {cmd_id}) from Master {master_idx} for LBA {lba}, {num_blocks} blocks.")
            
            # Simulate NVMe controller processing delay (before accessing SRAM)
            yield self.env.timeout(self.processing_latency_ns)
            
            status = 'SUCCESS'
            bytes_transferred = 0
            read_data_payload = None
            error_message = None
            
            try:
                if cmd_type == 'write':
                    sram_addr = lba * self.block_size_bytes
                    sram_size = num_blocks * self.block_size_bytes
                    
                    if sram_addr + sram_size > self.total_size_bytes or sram_addr < 0:
                        raise IndexError("Write address out of bounds for internal SRAM")
                    if len(data_payload) != sram_size:
                        raise ValueError("Data payload size mismatch for write command")
                        
                    # Create SRAM write packet
                    sram_write_packet = Packet(
                        id=cmd_id, # Use same ID for correlation
                        type='write',
                        source_id=f"NVMeSlave_{self.env.now}",
                        destination_id="InternalSRAM",
                        address=sram_addr,
                        size=sram_size,
                        data=data_payload
                    )
                    yield self.sram_req_channel.put((0, sram_write_packet)) # 0 is dummy master_idx for internal SRAM
                    sram_response = yield self.sram_resp_channel.get()
                    
                    if sram_response.status == 'OK':
                        bytes_transferred = len(data_payload)
                        print(f"[{self.env.now}] NVMeSlaveDevice: Wrote {bytes_transferred} bytes to LBA {lba} via internal SRAM.")
                    else:
                        status = 'ERROR'
                        error_message = f"Internal SRAM write error: {sram_response.error_message}"
                        
                elif cmd_type == 'read':
                    sram_addr = lba * self.block_size_bytes
                    sram_size = num_blocks * self.block_size_bytes
                    
                    if sram_addr + sram_size > self.total_size_bytes or sram_addr < 0:
                        raise IndexError("Read address out of bounds for internal SRAM")
                        
                    # Create SRAM read packet
                    sram_read_packet = Packet(
                        id=cmd_id,
                        type='read',
                        source_id=f"NVMeSlave_{self.env.now}",
                        destination_id="InternalSRAM",
                        address=sram_addr,
                        size=sram_size
                    )
                    yield self.sram_req_channel.put((0, sram_read_packet)) # 0 is dummy master_idx for internal SRAM
                    sram_response = yield self.sram_resp_channel.get()
                    
                    if sram_response.status == 'OK':
                        read_data_payload = sram_response.data
                        bytes_transferred = len(read_data_payload)
                        print(f"[{self.env.now}] NVMeSlaveDevice: Read {bytes_transferred} bytes from LBA {lba} via internal SRAM.")
                    else:
                        status = 'ERROR'
                        error_message = f"Internal SRAM read error: {sram_response.error_message}"
                        
                elif cmd_type == 'identify':
                    read_data_payload = b'NVMe_Controller_ID_Data'
                    bytes_transferred = len(read_data_payload)
                    print(f"[{self.env.now}] NVMeSlaveDevice: Processed Identify command.")

                else:
                    status = 'ERROR'
                    error_message = f"Unsupported command type: {cmd_type}"
                    print(f"[{self.env.now}] NVMeSlaveDevice: {error_message}")

            except (IndexError, ValueError, TypeError) as e:
                status = 'ERROR'
                error_message = str(e)
                print(f"[{self.env.now}] NVMeSlaveDevice Error: {error_message}")
            
            # Create completion Packet
            completion_packet = Packet(
                id=cmd_id,
                type='response',
                source_id=f"NVMeSlave_{self.env.now}",
                destination_id=request_packet.source_id,
                status=status,
                bytes_transferred=bytes_transferred,
                data=read_data_payload,
                error_message=error_message
            )
            
            yield self.out_port.put((master_idx, completion_packet)) # Pass the Packet object
            print(f"[{self.env.now}] NVMeSlaveDevice: Sent completion for command ID {cmd_id} with status {status}.")
