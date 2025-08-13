import simpy
from src.base.packet import Packet # New import

class Uart:
    def __init__(self, env, tx_port, rx_port, baud_rate=115200):
        self.env = env
        self.tx_port = tx_port # Port to receive data to be transmitted
        self.rx_port = rx_port # Port to send received data to
        self.bit_time_ns = 1e9 / baud_rate # Time to transmit one bit in ns
        self.action = env.process(self.run())

    def run(self):
        while True:
            # Wait for data to arrive on the transmit port (expecting a Packet)
            packet = yield self.tx_port.get()
            
            # Assume packet.data is bytes or can be converted to it
            data_to_transmit = packet.data if isinstance(packet.data, bytes) else str(packet.data).encode('utf-8')
            num_bits = len(data_to_transmit) * 8 # 8 bits per byte
            transmission_delay = num_bits * self.bit_time_ns
            
            print(f"[{self.env.now:.2f}] UART: Starting transmission of {len(data_to_transmit)} bytes for Packet ID {packet.id}.")
            
            # Simulate the time it takes to transmit the data
            yield self.env.timeout(transmission_delay)
            
            print(f"[{self.env.now:.2f}] UART: Finished transmission for Packet ID {packet.id}. Placing in RX port.")
            
            # Place the received data (original packet) into the rx_port for other components to read
            yield self.rx_port.put(packet) # Pass the original Packet object