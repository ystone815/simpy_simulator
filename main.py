import simpy
from components.cpu import Cpu
from components.memory import Memory
from components.bus import SoCBus
from components.uart import Uart
from components.nvme_slave import NVMeSlaveDevice # New import
from base.delay import DelayLine, DelayLineUtil

# 1. Setup Simulation Environment
env = simpy.Environment()

# 2. Create Communication Channels (simpy.Store instances)
# These represent wires/buses connecting components

# CPU <-> Bus Channels (Master 0)
cpu_req_bus_channel = simpy.Store(env)
bus_resp_cpu_channel = simpy.Store(env)

# NVMe Host Queue Manager <-> Bus Channels (Master 1)
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

# Slave components
memory = Memory(env, size_in_bytes=1024, 
                in_port=bus_req_memory_channel, out_port=memory_resp_bus_channel,
                access_latency_ns=5)

uart = Uart(env, tx_port=bus_req_uart_channel, rx_port=uart_resp_bus_channel)

nvme_slave = NVMeSlaveDevice(env, 
                             in_port=bus_req_nvme_channel, 
                             out_port=nvme_resp_bus_channel,
                             size_in_blocks=256, # 256 blocks * 4KB/block = 1MB
                             block_size_bytes=4096,
                             processing_latency_ns=500) # Simulate some latency

# Data Path Components (e.g., Crypto Engine)
crypto_engine = DelayLineUtil(env, 
                              in_port=crypto_in_channel, 
                              out_port=crypto_out_channel,
                              data_path_width_byte=16, 
                              clock_freq_mhz=200, 
                              utilization=0.8)

# SoC Bus - connects masters to slaves
soc_bus = SoCBus(env,
                 master_req_ports=[cpu_req_bus_channel, nvme_host_req_bus_channel], # Added NVMe Host as a master
                 master_resp_ports=[bus_resp_cpu_channel, bus_resp_nvme_host_channel], # Added NVMe Host response
                 slave_req_ports=[bus_req_memory_channel, bus_req_uart_channel, bus_req_nvme_channel], # Added NVMe Slave
                 slave_resp_ports=[memory_resp_bus_channel, uart_resp_bus_channel, nvme_resp_bus_channel]) # Added NVMe Slave response

# 4. Define Test Processes
def crypto_test(env, crypto_engine_in, crypto_engine_out):
    yield env.timeout(10)
    print(f"[{env.now}] Test: Sending 32-byte data to crypto engine.")
    yield crypto_engine_in.put(("data_block_1", 32))
    processed_data = yield crypto_engine_out.get()
    print(f"[{env.now}] Test: Received processed data: {processed_data}")
    
    yield env.timeout(5)
    print(f"[{env.now}] Test: Sending 128-byte data to crypto engine.")
    yield crypto_engine_in.put(("data_block_2", 128))
    processed_data = yield crypto_engine_out.get()
    print(f"[{env.now}] Test: Received processed data: {processed_data}")

def nvme_traffic_generator(env, req_port, resp_port, nvme_slave_idx):
    yield env.timeout(20) # Wait a bit for CPU to do its thing
    print(f"[{env.now}] NVMe Host: Starting traffic generation.")

    # Command 1: Identify (cmd_id, cmd_type, lba, num_blocks, data_payload)
    cmd_id = 1
    nvme_cmd = (cmd_id, 'identify', 0, 0, b'')
    print(f"[{env.now}] NVMe Host: Sending Identify command (ID: {cmd_id}).")
    yield req_port.put((nvme_slave_idx, nvme_cmd)) # (target_slave_idx, actual_command)
    response = yield resp_port.get()
    print(f"[{env.now}] NVMe Host: Received response for ID {cmd_id}: {response}")
    assert response[0] == cmd_id and response[1] == 'SUCCESS'

    yield env.timeout(50)

    # Command 2: Write 1 block to LBA 0
    cmd_id = 2
    write_data = b'\xAA' * nvme_slave.block_size_bytes # Fill with AA
    nvme_cmd = (cmd_id, 'write', 0, 1, write_data)
    print(f"[{env.now}] NVMe Host: Sending Write command (ID: {cmd_id}) to LBA 0.")
    yield req_port.put((nvme_slave_idx, nvme_cmd))
    response = yield resp_port.get()
    print(f"[{env.now}] NVMe Host: Received response for ID {cmd_id}: {response}")
    assert response[0] == cmd_id and response[1] == 'SUCCESS'

    yield env.timeout(50)

    # Command 3: Read 1 block from LBA 0
    cmd_id = 3
    nvme_cmd = (cmd_id, 'read', 0, 1, b'')
    print(f"[{env.now}] NVMe Host: Sending Read command (ID: {cmd_id}) from LBA 0.")
    yield req_port.put((nvme_slave_idx, nvme_cmd))
    response = yield resp_port.get()
    print(f"[{env.now}] NVMe Host: Received response for ID {cmd_id}: {response}")
    assert response[0] == cmd_id and response[1] == 'SUCCESS' and response[3] == write_data

    yield env.timeout(50)

    # Command 4: Write 2 blocks to LBA 10 (out of bounds for 256 blocks)
    cmd_id = 4
    write_data_large = b'\xBB' * (2 * nvme_slave.block_size_bytes)
    nvme_cmd = (cmd_id, 'write', 255, 2, write_data_large) # LBA 255 + 2 blocks will be out of bounds
    print(f"[{env.now}] NVMe Host: Sending Write command (ID: {cmd_id}) to LBA 255 (out of bounds test).")
    yield req_port.put((nvme_slave_idx, nvme_cmd))
    response = yield resp_port.get()
    print(f"[{env.now}] NVMe Host: Received response for ID {cmd_id}: {response}")
    assert response[0] == cmd_id and response[1] == 'ERROR'


# 5. Run Simulation
print("--- SoC Simulation Started ---")

# Start test processes
env.process(crypto_test(env, crypto_in_channel, crypto_out_channel))
# Note: NVMe Slave is Slave 2 in the soc_bus slave_req_ports list (0: Memory, 1: UART, 2: NVMe)
env.process(nvme_traffic_generator(env, nvme_host_req_bus_channel, bus_resp_nvme_host_channel, nvme_slave_idx=2))

env.run(until=1000) # Run for a longer time to see more interactions
print("--- SoC Simulation Finished ---")

# Print utilization of crypto engine
print(f"\nCrypto Engine Utilization: {crypto_engine.get_utilization():.2f}%")
