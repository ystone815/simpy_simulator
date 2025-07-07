import simpy
from base.packet import Packet # New import

class SoCBus:
    def __init__(self, env, master_req_ports, master_resp_ports, slave_req_ports, slave_resp_ports):
        self.env = env
        self.master_req_ports = master_req_ports # List of simpy.Store for master requests
        self.master_resp_ports = master_resp_ports # List of simpy.Store for master responses
        self.slave_req_ports = slave_req_ports # List of simpy.Store for slave requests
        self.slave_resp_ports = slave_resp_ports # List of simpy.Store for slave responses

        # Start processes for handling requests and responses
        self.action = env.process(self.run())

    def run(self):
        # This process handles requests from masters and routes them to slaves
        self.env.process(self._handle_master_requests())
        # This process handles responses from slaves and routes them back to masters
        self.env.process(self._handle_slave_responses())
        
        print(f"[{self.env.now}] SoCBus: Running...")
        # The main run loop can just yield forever as sub-processes handle the work
        yield self.env.timeout(1) # Just to start the processes, then they run independently
        while True:
            yield self.env.timeout(1000) # Keep the bus process alive

    def _handle_master_requests(self):
        while True:
            get_events = [port.get() for port in self.master_req_ports]
            result = yield self.env.any_of(get_events)
            
            master_idx = -1
            request_tuple = None # This will be (slave_idx, Packet)
            for i, port in enumerate(self.master_req_ports):
                if port.get() in result:
                    master_idx = i
                    request_tuple = result[port.get()]
                    break
            
            if request_tuple:
                slave_idx, request_packet = request_tuple # Unpack the tuple
                
                if not isinstance(request_packet, Packet):
                    print(f"[{self.env.now}] SoCBus Error: Received non-Packet object from Master {master_idx}: {request_packet}")
                    # Send error back to master
                    error_packet = Packet(id=-1, type='error', source_id='SoCBus', destination_id=f"Master_{master_idx}", error_message="Invalid request format: Not a Packet object")
                    yield self.master_resp_ports[master_idx].put(error_packet)
                    continue

                print(f"[{self.env.now}] SoCBus: Routing {request_packet.type} packet (ID: {request_packet.id}) from Master {master_idx} to Slave {slave_idx}.")
                
                if 0 <= slave_idx < len(self.slave_req_ports):
                    # Prepend master_idx to the request so slave knows where to send response
                    yield self.slave_req_ports[slave_idx].put((master_idx, request_packet))
                else:
                    print(f"[{self.env.now}] SoCBus Error: Invalid slave index {slave_idx} from Master {master_idx} for packet ID {request_packet.id}.")
                    # Send error back to master
                    error_packet = Packet(id=request_packet.id, type='response', source_id='SoCBus', destination_id=request_packet.source_id, status='ERROR', error_message="Invalid Slave Index")
                    yield self.master_resp_ports[master_idx].put(error_packet)
            else:
                print(f"[{self.env.now}] SoCBus Error: Could not identify requesting master.")

    def _handle_slave_responses(self):
        while True:
            get_events = [port.get() for port in self.slave_resp_ports]
            result = yield self.env.any_of(get_events)
            
            slave_idx = -1
            response_tuple = None # This will be (master_idx, Packet)
            for i, port in enumerate(self.slave_resp_ports):
                if port.get() in result:
                    slave_idx = i
                    response_tuple = result[port.get()]
                    break
            
            if response_tuple:
                master_idx, response_packet = response_tuple # Unpack the tuple
                
                if not isinstance(response_packet, Packet):
                    print(f"[{self.env.now}] SoCBus Error: Received non-Packet object from Slave {slave_idx}: {response_packet}")
                    # This error cannot be routed back easily without knowing the original master
                    continue

                print(f"[{self.env.now}] SoCBus: Routing {response_packet.type} packet (ID: {response_packet.id}) from Slave {slave_idx} to Master {master_idx}.")
                
                if 0 <= master_idx < len(self.master_resp_ports):
                    yield self.master_resp_ports[master_idx].put(response_packet) # Pass the Packet object directly
                else:
                    print(f"[{self.env.now}] SoCBus Error: Invalid master index {master_idx} from Slave {slave_idx} for packet ID {response_packet.id}.")
            else:
                print(f"[{self.env.now}] SoCBus Error: Could not identify responding slave.")
