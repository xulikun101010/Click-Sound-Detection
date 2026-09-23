from multiprocessing import Process
from Zetong_old_code.log_writer0 import init_log, write_log
import subprocess
import time
from datetime import datetime

from robot_interface_wrapper import ROB_INF as ROB_INF


# ============================================================
# Parameters
# ============================================================

# Force drop threshold
#
# If change <= -3.0:
# contact / insertion event is detected.
FORCE_DROP_THRESHOLD = 3.0

# Force monitoring interval
# 0.01 s = approximately 10 ms
FORCE_SAMPLE_INTERVAL = 0.01


# ============================================================
# Run external script
# ============================================================

def run_script(script_name):

    subprocess.run(
        [
            "python",
            script_name
        ]
    )


# ============================================================
# Robot connection
# ============================================================

def connect(ip_address):

    # Initialize all basic method data

    if ip_address == "":

        raise Exception(
            "Robot IP not defined"
        )

    RI = ROB_INF(
        ip_address
    )

    RI.connect()

    return RI


# ============================================================
# Detection program
# ============================================================

def detection_program():

    """
    Detection program.

    It starts when Flag 5 is triggered
    and runs for 5 seconds.
    """

    print(
        "Detection program started..."
    )

    time.sleep(
        5
    )

    print(
        "Detection program ended."
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # Initialize CSV log
    # ========================================================

    init_log()

    # ========================================================
    # Sub-scripts
    # ========================================================

    scripts = [

        "EConMateSense/record_signal_waveform.py",

    ]

    # ========================================================
    # Robot connection
    # ========================================================

    r2_ip = "192.168.1.100"

    r2_ri = connect(
        r2_ip
    )

    # ========================================================
    # Process / State Variables
    # ========================================================

    processes = []

    # Indicates whether recording/detection programs
    # have already been started
    scripts_started = False

    # Ensures that only one force value is logged
    # for one insertion process
    connect_logged = False

    # Previous force[2] value
    previous_force_z = None

    try:

        while True:

            # =================================================
            # Read Robot Data
            # =================================================

            flag = r2_ri.get_flag(
                5,
                0
            )

            force = r2_ri.get_force()

            current_force_z = force[
                2
            ]

            # Optional real-time display
            print(
                "Flag:",
                flag,
                "| Force[2]:",
                round(
                    current_force_z,
                    2
                )
            )

            # =================================================
            # FLAG 5 TRIGGER
            # =================================================

            if (
                flag
                and
                not scripts_started
            ):

                print(
                    "Flag 5 triggered, "
                    "starting detection program "
                    "and all sub-scripts."
                )

                # ---------------------------------------------
                # Start detection program
                # ---------------------------------------------

                detection_process = Process(
                    target=detection_program
                )

                detection_process.start()

                # ---------------------------------------------
                # Start recording / other scripts
                # ---------------------------------------------

                for script in scripts:

                    p = Process(
                        target=run_script,
                        args=(
                            script,
                        )
                    )

                    p.start()

                    processes.append(
                        p
                    )

                # ---------------------------------------------
                # Start Force Monitoring
                #
                # Important:
                # We do NOT write Force to log here.
                #
                # This value becomes the first reference value
                # for detecting the later sudden force drop.
                # ---------------------------------------------

                previous_force_z = (
                    current_force_z
                )

                connect_logged = False

                scripts_started = True

                print(
                    "Force monitoring started."
                )

                print(
                    "Initial Force[2]:",
                    round(
                        current_force_z,
                        2
                    )
                )

            # =================================================
            # FORCE DROP MONITORING
            # =================================================

            elif scripts_started:

                # We need a previous force value
                if previous_force_z is not None:

                    force_change = (

                        current_force_z
                        -
                        previous_force_z
                    )

                    # -----------------------------------------
                    # Print change for testing
                    # -----------------------------------------

                    print(
                        "Force change:",
                        round(
                            force_change,
                            2
                        )
                    )

                    # -----------------------------------------
                    # Detect sudden decrease
                    # -----------------------------------------

                    if (
                        not connect_logged
                        and
                        force_change
                        <=
                        -FORCE_DROP_THRESHOLD
                    ):

                        print(
                            "======================================"
                        )

                        print(
                            "FORCE DROP DETECTED"
                        )

                        print(
                            "Previous Force[2]:",
                            round(
                                previous_force_z,
                                2
                            )
                        )

                        print(
                            "Current Force[2]:",
                            round(
                                current_force_z,
                                2
                            )
                        )

                        print(
                            "Force Change:",
                            round(
                                force_change,
                                2
                            )
                        )

                        print(
                            "Writing current Force[2] to log..."
                        )

                        # -------------------------------------
                        # Write instantaneous force value
                        # AFTER the sudden decrease
                        # -------------------------------------

                        write_log(

                            noise_env=
                                "noise level Low",

                            interface=
                                "SchaltBedb",

                            force_third=
                                round(
                                    current_force_z,
                                    2
                                )
                        )

                        connect_logged = True

                        print(
                            "Force value written:",
                            round(
                                current_force_z,
                                2
                            )
                        )

                        print(
                            "======================================"
                        )

                # ---------------------------------------------
                # Update previous value
                # ---------------------------------------------

                previous_force_z = (
                    current_force_z
                )

            # =================================================
            # Reset after Flag 5 becomes inactive
            # =================================================

            if (
                not flag
                and
                scripts_started
            ):

                print(
                    "Flag 5 reset. "
                    "Preparing for next insertion."
                )

                scripts_started = False

                connect_logged = False

                previous_force_z = None

                # Stop running subprocesses
                for p in processes:

                    if p.is_alive():

                        p.terminate()

                processes.clear()

            # =================================================
            # Sampling Interval
            # =================================================

            time.sleep(
                FORCE_SAMPLE_INTERVAL
            )

    except KeyboardInterrupt:

        print(
            "\nProgram manually terminated, "
            "closing all subprocesses..."
        )

        for p in processes:

            if p.is_alive():

                p.terminate()

        print(
            "All subprocesses have been terminated. "
            "Please check detection_log.csv"
        )