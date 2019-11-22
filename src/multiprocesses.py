"""
Multiprocessing -> Spawn multiprocesses.
Allows to write concurrent programs by sidestepping Global Interpretor Lock.
GIL forces python to execute one thread at a time. Multiprocessing bypasses this.
"""

import os
from multiprocessing import Process, current_process


def square(numbers):
    for number in numbers:

        result = number * number
        # We can use the 'os' module to print the process id asssigned to this function by the os
        # process_id = os.getpid()
        # print(f"process_id {process_id}")

        process_name = current_process().name
        print(f"Process name {process_name}")
        print(f"The number {number} squares to {result}.")

if __name__ =='__main__':
    processes = []
    #numbers = [1, 2, 3, 4]
    numbers = range(1000)
    # Spawn 50 processes
    for _ in range(50):
        process = Process(target=square, args = (numbers,))
        #square(number)
        processes.append(process)
        # Processes are spawned by creating a process object and then calling its start method
        process.start()


    # make use of .join method to make sure that all processes have finished before we run any further code
    for process in processes:
        process.join()

    print("Multiprocessing Done ") 
