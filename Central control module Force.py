from multiprocessing import Process
from Zetong_old_code.log_writer0 import init_log, write_log

import subprocess
import time
import os

import matplotlib.pyplot as plt

from robot_interface_wrapper import ROB_INF as ROB_INF


# ============================================================
# Parameters
# ============================================================

FORCE_DROP_THRESHOLD = 3.0

FORCE_SAMPLE_INTERVAL = 0.01

FORCE_RECORD_DURATION = 5.0

FORCE_FIGURE_DIR = (
    r"D:\AHr\Acoustic\data\recording\force figure"
)

os.makedirs(
    FORCE_FIGURE_DIR,
    exist_ok=True
)


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
# Get next force figure number
# ============================================================

def get_next_force_figure_path():

    existing_files = [
        filename
        for filename in os.listdir(
            FORCE_FIGURE_DIR
        )
        if filename.endswith(
            "_force.png"
        )
    ]

    indices = []

    for filename in existing_files:

        try:

            index_text = filename.split(
                "_"
            )[0]

            index = int(
                index_text
            )

            indices.append(
                index
            )

        except ValueError:

            pass

    if len(indices) > 0:

        next_index = max(
            indices
        ) + 1

    else:

        next_index = 1

    filename = (
        f"{next_index:03d}_force.png"
    )

    save_path = os.path.join(
        FORCE_FIGURE_DIR,
        filename
    )

    return save_path


# ============================================================
# Save force curve
# ============================================================

def save_force_figure(
    force_times,
    force_values
):

    if len(force_times) == 0:

        print(
            "No force data available. "
            "Force figure was not saved."
        )

        return

    save_path = (
        get_next_force_figure_path()
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        force_times,
        force_values
    )

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Force[2]"
    )

    plt.title(
        "Force Curve During Insertion"
    )

    plt.grid(
        True
    )

    plt.xlim(
        0,
        FORCE_RECORD_DURATION
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()

    print(
        "======================================"
    )

    print(
        "Force figure saved:"
    )

    print(
        save_path
    )

    print(
        "Number of force samples:",
        len(force_values)
    )

    print(
        "======================================"
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

    scripts_started = False

    connect_logged = False

    previous_force_z = None

    # ========================================================
    # Force curve variables
    # ========================================================

    force_times = []

    force_values = []

    force_record_start = None

    force_curve_saved = False

    # ========================================================
    # Display initial force once
    # ========================================================

    initial_force = (
        r2_ri.get_force()
    )

    initial_force_z = (
        initial_force[2]
    )

    print(
        "Initial Force[2]:",
        round(
            initial_force_z,
            2
        )
    )

    print(
        "Waiting for Flag 5..."
    )

    try:

        while True:

            # =================================================
            # Read Robot Data
            # =================================================

            flag = r2_ri.get_flag(
                5,
                0
            )

            force = (
                r2_ri.get_force()
            )

            current_force_z = (
                force[2]
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
                    "\n======================================"
                )

                print(
                    "Flag 5 triggered."
                )

                print(
                    "Starting detection program "
                    "and recording."
                )

                print(
                    "======================================"
                )

                # ---------------------------------------------
                # Start detection program
                # ---------------------------------------------

                detection_process = Process(
                    target=detection_program
                )

                detection_process.start()

                # ---------------------------------------------
                # Start recording scripts
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
                # Start force monitoring
                # ---------------------------------------------

                previous_force_z = (
                    current_force_z
                )

                connect_logged = False

                scripts_started = True

                # ---------------------------------------------
                # Start 5-second force curve recording
                # ---------------------------------------------

                force_times = []

                force_values = []

                force_record_start = (
                    time.time()
                )

                force_curve_saved = False

                # First point = t = 0
                force_times.append(
                    0.0
                )

                force_values.append(
                    current_force_z
                )

                print(
                    "Force monitoring started."
                )

                print(
                    "Initial Force[2] at Flag 5:",
                    round(
                        current_force_z,
                        2
                    )
                )

            # =================================================
            # FORCE MONITORING AFTER FLAG 5
            # =================================================

            elif scripts_started:

                # ---------------------------------------------
                # Real-time display
                # ---------------------------------------------

                print(
                    "Force[2]:",
                    round(
                        current_force_z,
                        2
                    )
                )

                # ---------------------------------------------
                # Record force curve for first 5 seconds
                # ---------------------------------------------

                if (
                    force_record_start
                    is not None
                    and
                    not force_curve_saved
                ):

                    elapsed_time = (
                        time.time()
                        -
                        force_record_start
                    )

                    if (
                        elapsed_time
                        <=
                        FORCE_RECORD_DURATION
                    ):

                        force_times.append(
                            elapsed_time
                        )

                        force_values.append(
                            current_force_z
                        )

                    else:

                        save_force_figure(
                            force_times,
                            force_values
                        )

                        force_curve_saved = True

                        print(
                            "5-second force recording finished."
                        )

                # ---------------------------------------------
                # Force drop monitoring
                # ---------------------------------------------

                if previous_force_z is not None:

                    force_change = (

                        current_force_z
                        -
                        previous_force_z
                    )

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
                # Update previous force value
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

                # ---------------------------------------------
                # Save force curve if Flag 5 resets
                # before normal 5-second save
                # ---------------------------------------------

                if (
                    not force_curve_saved
                    and
                    len(force_times) > 0
                ):

                    save_force_figure(
                        force_times,
                        force_values
                    )

                    force_curve_saved = True

                print(
                    "Flag 5 reset. "
                    "Preparing for next insertion."
                )

                scripts_started = False

                connect_logged = False

                previous_force_z = None

                force_record_start = None

                force_times = []

                force_values = []

                force_curve_saved = False

                # ---------------------------------------------
                # Stop running subprocesses
                # ---------------------------------------------

                for p in processes:

                    if p.is_alive():

                        p.terminate()

                processes.clear()

                print(
                    "Force display stopped."
                )

                print(
                    "Waiting for Flag 5..."
                )

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

        # -----------------------------------------------------
        # Save unfinished force curve
        # -----------------------------------------------------

        if (
            scripts_started
            and
            not force_curve_saved
            and
            len(force_times) > 0
        ):

            save_force_figure(
                force_times,
                force_values
            )

        for p in processes:

            if p.is_alive():

                p.terminate()

        print(
            "All subprocesses have been terminated. "
            "Please check detection_log.csv"
        )