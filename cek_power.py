import os
import time
import glob

def find_power_sensor():
    # Search for all hwmon (Hardware Monitor) directories
    hwmon_paths = glob.glob("/sys/class/hwmon/hwmon*")
    
    sensor_path = None
    
    print(f"Searching for power sensors in {len(hwmon_paths)} hwmon locations...")
    
    for path in hwmon_paths:
        try:
            # Check sensor name
            with open(os.path.join(path, "name"), "r") as f:
                name = f.read().strip()
            
            # Orin Nano typically uses an INA3221 sensor on a specific bus
            # We look for files related to 'power' or 'in' (input voltage/current)
            # Usually 'in1_input' or 'power1_input' corresponds to VDD_IN (Total Power)
            
            # Check if there are power-related files in this directory
            files = os.listdir(path)
            # Power-related file patterns are usually in1_input, power1_input, etc.
            # On Orin Nano, VDD_IN is often mapped to channel 1
            
            if "ina3221" in name:
                print(f"-> INA3221 sensor found at: {path}")
                
                # Check label to confirm this is VDD_IN
                # Sometimes the label is stored in in1_label
                if os.path.exists(os.path.join(path, "in1_label")):
                    with open(os.path.join(path, "in1_label"), "r") as fl:
                        label = fl.read().strip()
                    print(f"   Channel 1 Label: {label}")
                
                # Assume this is the correct path if none has been selected yet
                if sensor_path is None:
                    sensor_path = path
                    
        except Exception:
            continue
            
    return sensor_path

def read_file_value(path):
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except:
        return 0

def monitor_orin():
    base_path = find_power_sensor()
    
    if not base_path:
        print("\n[ERROR] Unable to find INA3221 power sensor.")
        print("Make sure you are running this script with 'sudo'.")
        print("If it still fails, your JetPack kernel may restrict i2c access.")
        return

    print(f"\nPower Monitoring Active from: {base_path}")
    print("Note: Orin Nano usually ONLY allows reading Total Power (VDD_IN).")
    print("-" * 40)

    try:
        while True:
            # Try reading Power directly (mW). File name may be 'power1_input'
            # On JetPack 6, 'power1_input' is often unavailable, so we calculate manually
            # VDD_IN is usually mapped to channel 1
            
            voltage_mv = read_file_value(os.path.join(base_path, "in1_input"))
            current_ma = read_file_value(os.path.join(base_path, "curr1_input"))
            
            # Sometimes 'power1_input' exists and already provides mW
            power_mw_direct = read_file_value(os.path.join(base_path, "power1_input"))
            
            # Select final power value
            if power_mw_direct > 0:
                final_power = power_mw_direct
            else:
                # Manual calculation: P = V × I / 1000
                # (mV × mA = µW, divide by 1000 to get mW)
                final_power = (voltage_mv * current_ma) / 1000

            print(f"Total Power (VDD_IN): {int(final_power)} mW  (Voltage: {voltage_mv} mV)")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    monitor_orin()
