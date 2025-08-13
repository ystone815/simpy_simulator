import simpy
from components.cpu import Cpu
from components.memory import Memory
from components.bus import SoCBus
from components.uart import Uart
from components.nvme_slave import NVMeSlaveDevice
from components.nvme_host import NVMeHost
from base.delay import DelayLine, DelayLineUtil
from base.packet import Packet # New import

# 1. Setup Simulation Environment
env = simpy.Environment()

# 2. Create Communication Channels (simpy.Store instances)
# These represent wires/buses connecting components

# CPU <-> Bus Channels (Master 0)
cpu_req_bus_channel = simpy.Store(env)
bus_resp_cpu_channel = simpy.Store(env)

# NVMe Host <-> Bus Channels (Master 1)
nvme_host_req_bus_channel = simpy.Store(env)
bus_resp_nvme_host_channel = simpy.Store(env)

# Memory <-> Bus Channels (Slave 0)
bus_req_memory_channel = simpy.Store(env)
memory_resp_bus_channel = simpy.Store(env)

# UART <-> Bus Channels (Slave 1)
bus_req_uart_channel = simpy.Store(env)
uart_resp_bus_channel = simpy.Store(env)

# NVMe Slave Device <-> Bus Channels (Slave 2)
bus_req_nvme_channel = simpy.Store(env)
nvme_resp_bus_channel = simpy.Store(env)

# Example: DelayLineUtil for a specific data path (e.g., crypto engine)
crypto_in_channel = simpy.Store(env)
crypto_out_channel = simpy.Store(env)


# 3. Create HW Components and connect them via channels
# Master components
cpu = Cpu(env, req_port=cpu_req_bus_channel, resp_port=bus_resp_cpu_channel)

# NVMe Host (Master 1)
# Note: NVMe Slave is Slave 2 in the soc_bus slave_req_ports list (0: Memory, 1: UART, 2: NVMe)
nvme_host = NVMeHost(env, 
                     req_port=nvme_host_req_bus_channel, 
                     resp_port=bus_resp_nvme_host_channel, 
                     nvme_slave_idx=2) 

# Slave components
memory = Memory(env, size_in_bytes=1024, 
                in_port=bus_req_memory_channel, out_port=memory_resp_bus_channel,
                access_latency_ns=5)

uart = Uart(env, tx_port=bus_req_uart_channel, rx_port=uart_resp_bus_channel)

nvme_slave = NVMeSlaveDevice(env, 
                             in_port=bus_req_nvme_channel, 
                             out_port=nvme_resp_bus_channel,
                             # size_in_blocks=256, # Removed
                             # block_size_bytes=4096, # Removed
                             processing_latency_ns=500)

# Data Path Components (e.g., Crypto Engine)
crypto_engine = DelayLineUtil(env, 
                              in_port=crypto_in_channel, 
                              out_port=crypto_out_channel,
                              data_path_width_byte=16, 
                              clock_freq_mhz=200, 
                              utilization=0.8)

# SoC Bus - connects masters to slaves
soc_bus = SoCBus(env,
                 master_req_ports=[cpu_req_bus_channel, nvme_host_req_bus_channel], 
                 master_resp_ports=[bus_resp_cpu_channel, bus_resp_nvme_host_channel], 
                 slave_req_ports=[bus_req_memory_channel, bus_req_uart_channel, bus_req_nvme_channel], 
                 slave_resp_ports=[memory_resp_bus_channel, uart_resp_bus_channel, nvme_resp_bus_channel])

# 4. Define Test Processes
def crypto_test(env, crypto_engine_in, crypto_engine_out):
    yield env.timeout(10)
    
    # Send 32-byte data
    packet_id = 100
    data_size = 32
    crypto_packet_1 = Packet(
        id=packet_id,
        type='data',
        source_id='CryptoTest',
        destination_id='CryptoEngine',
        size=data_size,
        data=b'\xAA' * data_size,
        timestamp=env.now
    )
    print(f"[{env.now}] Test: Sending {data_size}-byte data to crypto engine (ID: {packet_id}).")
    yield crypto_engine_in.put(crypto_packet_1)
    processed_packet_1 = yield crypto_engine_out.get()
    print(f"[{env.now}] Test: Received processed data for ID {processed_packet_1.id}: {processed_packet_1.data[:10]}...")
    
    yield env.timeout(5)
    
    # Send 128-byte data
    packet_id = 101
    data_size = 128
    crypto_packet_2 = Packet(
        id=packet_id,
        type='data',
        source_id='CryptoTest',
        destination_id='CryptoEngine',
        size=data_size,
        data=b'\xBB' * data_size,
        timestamp=env.now
    )
    print(f"[{env.now}] Test: Sending {data_size}-byte data to crypto engine (ID: {packet_id}).")
    yield crypto_engine_in.put(crypto_packet_2)
    processed_packet_2 = yield crypto_engine_out.get()
    print(f"[{env.now}] Test: Received processed data for ID {processed_packet_2.id}: {processed_packet_2.data[:10]}...")


# 5. Run Simulation
print("--- SoC Simulation Started ---")

# Start test processes
env.process(crypto_test(env, crypto_in_channel, crypto_out_channel))

env.run(until=1000) # Run for a longer time to see more interactions
print("--- SoC Simulation Finished ---")

# Print utilization of crypto engine
print(f"\nCrypto Engine Utilization: {crypto_engine.get_utilization():.2f}%")