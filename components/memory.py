import simpy
from base.packet import Packet # New import

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
            # Requests are expected to come from the SoCBus, which prepends master_idx
            master_idx, request_packet = yield self.in_port.get() # Get Packet object
            
            cmd_id = request_packet.id
            cmd_type = request_packet.type
            addr = request_packet.address
            size = request_packet.size
            data_payload = request_packet.data
            
            print(f"[{self.env.now}] Memory: Received {cmd_type} command (ID: {cmd_id}) from Master {master_idx} for Addr {addr}, Size {size}.")
            
            yield self.env.timeout(self.access_latency_ns)
            
            status = 'SUCCESS'
            read_data_payload = None
            error_message = None
            
            try:
                if cmd_type == 'write':
                    if not isinstance(data_payload, (bytes, bytearray)):
                        raise TypeError("Write data must be bytes or bytearray")
                    
                    if addr + len(data_payload) > self.size or addr < 0:
                        raise IndexError("Write address out of bounds")
                        
                    self.data[addr:addr + len(data_payload)] = data_payload
                    
                elif cmd_type == 'read':
                    if not isinstance(size, int) or size <= 0:
                        raise TypeError("Read size must be a positive integer")

                    if addr + size > self.size or addr < 0:
                        raise IndexError("Read address out of bounds")
                        
                    read_data_payload = self.data[addr:addr + size]
                    
                else:
                    status = 'ERROR'
                    error_message = f"Unsupported operation: {cmd_type}"

            except (ValueError, IndexError, TypeError) as e:
                status = 'ERROR'
                error_message = str(e)
                print(f"[{self.env.now}] Memory Error: {error_message}")
            
            # Create response Packet
            response_packet = Packet(
                id=cmd_id,
                type='response',
                source_id=f"Memory",
                destination_id=request_packet.source_id, # Respond to the original source
                status=status,
                data=read_data_payload,
                error_message=error_message
            )
            
            # Send response back to the master via the bus
            yield self.out_port.put((master_idx, response_packet)) # Pass the Packet object
            print(f"[{self.env.now}] Memory: Sent response for command ID {cmd_id} with status {status}.")