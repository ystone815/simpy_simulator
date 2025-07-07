import simpy

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
        # Use simpy.AnyOf to listen to all master request ports
        while True:
            get_events = [port.get() for port in self.master_req_ports]
            result = yield self.env.any_of(get_events)
            
            # Find which master sent the request
            master_idx = -1
            request = None
            for i, port in enumerate(self.master_req_ports):
                if port.get() in result: # Check if this port's get event was triggered
                    master_idx = i
                    request = result[port.get()]
                    break
            
            if request:
                # Request format: (slave_idx, original_request_tuple)
                # The first element of the request is assumed to be the target slave index
                slave_idx, original_request = request
                
                if 0 <= slave_idx < len(self.slave_req_ports):
                    print(f"[{self.env.now}] SoCBus: Routing request from Master {master_idx} to Slave {slave_idx}: {original_request}")
                    # Prepend master_idx to the request so slave knows where to send response
                    yield self.slave_req_ports[slave_idx].put((master_idx, original_request))
                else:
                    print(f"[{self.env.now}] SoCBus Error: Invalid slave index {slave_idx} from Master {master_idx}")
                    # Send error back to master
                    yield self.master_resp_ports[master_idx].put(('ERROR', 'Invalid Slave Index'))
            else:
                print(f"[{self.env.now}] SoCBus Error: Could not identify requesting master.")

    def _handle_slave_responses(self):
        # Use simpy.AnyOf to listen to all slave response ports
        while True:
            get_events = [port.get() for port in self.slave_resp_ports]
            result = yield self.env.any_of(get_events)
            
            # Find which slave sent the response
            slave_idx = -1
            response = None
            for i, port in enumerate(self.slave_resp_ports):
                if port.get() in result:
                    slave_idx = i
                    response = result[port.get()]
                    break
            
            if response:
                # Response format: (master_idx, original_response_tuple)
                # The first element of the response is assumed to be the target master index
                master_idx, original_response = response
                
                if 0 <= master_idx < len(self.master_resp_ports):
                    print(f"[{self.env.now}] SoCBus: Routing response from Slave {slave_idx} to Master {master_idx}: {original_response}")
                    yield self.master_resp_ports[master_idx].put(original_response)
                else:
                    print(f"[{self.env.now}] SoCBus Error: Invalid master index {master_idx} from Slave {slave_idx}")
            else:
                print(f"[{self.env.now}] SoCBus Error: Could not identify responding slave.")