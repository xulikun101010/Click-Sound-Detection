from multiprocessing import Process
from Zetong_old_code.log_writer0 import init_log, write_log
import subprocess
import time
from datetime import datetime

from robot_interface_wrapper import ROB_INF as ROB_INF


def run_script(script_name):
    subprocess.run(["python", script_name])


def connect(ip_address):
    # Initialize all basic method data
    if ip_address == "":
        raise Exception("Robot IP not defined")
    RI = ROB_INF(ip_address)
    RI.connect()
    return RI


def detection_program():
    """
    Simulate the detection program.
    This function is only started when the flag is (True, 1) and runs for 5 seconds.
    Replace this logic with your actual detection code if needed.
    """
    print("Detection program started...")
    time.sleep(5)  # Detection program runs for 5 seconds
    print("Detection program ended.")


if __name__ == "__main__":
    # Initialize the CSV log file
    init_log()

    # Define the list of sub-scripts to run
    scripts = [
        "EConMateSense/record_signal_waveform.py",
    ]
    r2_ip = "192.168.1.100"
    r2_ri = connect(r2_ip)

    # Variables to manage subprocesses and startup status
    processes = []
    scripts_started = False  # Flag to indicate whether the sub-scripts have been started
    connect_logged = False  # Flag to avoid repeated logging of "connecting"

    try:
        while True:
            flag = r2_ri.get_flag(5, 0)
            force = r2_ri.get_force()
            
            # Only start all programs when flag is (True, 0) and sub-scripts have not been started yet
            if flag and not scripts_started:
                print("Flag is (True, 1), starting detection program and all sub-scripts.")
                # Start the detection program (runs for 5 seconds)
                detection_process = Process(target=detection_program)
                detection_process.start()

                # Write to CSV log after the detection program starts
                # log_message = f"Programs started at {datetime.now()}"
                # write_log("CNN", "Noise Environment", log_message)

                # Start all sub-scripts
                for script in scripts:
                    p = Process(target=run_script, args=(script,))
                    p.start()
                    processes.append(p)

                if not connect_logged:
                    print("WRITE LOG: TRIGGERED", force[2])
                    write_log(
                        noise_env="noise level llow",  # Placeholder value, replace with actual noise environment if available
                        interface="LSRegler", # Interface name, replace with actual interface if available
                        force_third=round(force[2],2)
                        )
                    connect_logged = True
                elif force[2] >= 140:
                    # Reset the flag if the force value recovers
                    connect_logged = False

                scripts_started = True

            # When flag is (True, 0), do not start any programs and reset startup status
            elif flag == (True, 1):
                print("Flag is (True, 1), no programs will be started.")
                scripts_started = False

                for p in processes:
                    p.terminate()

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nProgram manually terminated, closing all subprocesses...")
        for p in processes:
            p.terminate()
            
        print("All subprocesses have been terminated. Please check detection_log.csv")
