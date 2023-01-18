import subprocess
import os
import signal
from pathlib import Path
import time
import shutil
from argparse import ArgumentParser
from pathlib import Path

"""
This file can have one of 3 contents:
    1. CleanSlate
    2. RetrievalStarted
    3. RetrievalComplete

Whenever the main process is started or restarted, the file contents should automatically change to Point 1 above.
If the file content is Retrieval Started(State 02) for more than n seconds, pre-empt the main process and restart it.

"""

def start_proc(out_command,out_file_path):
    # Can't use shell=True if you want the pid of `du`, not the
    # shell, so we have to do the redirection to file ourselves
    out_file_path = Path(out_file_path).parent
    proc = subprocess.Popen(out_command, stdout=open(os.path.join(out_file_path,"log.data"), "a"))
    print("Started Process with PID:", proc.pid)
    return proc.pid

def kill_proc(cur_pid):
    os.kill(cur_pid, signal.SIGTERM)
    print('Killed Process with PID: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ', cur_pid)
    time.sleep(25)

def get_retrieval_state(f_path):
    _content = open(f_path, 'r').read()
    _content = _content.lstrip('\n').rstrip('\n')
    return _content


def make_retrieval_dt(f_path, reset=True):
    
    if os.path.exists(f_path):
        with open(f_path, 'w') as f:
            f.write('CleanSlate')

    else:
        f_path_path = Path(f_path)
        print(f_path_path.parent.absolute())
        
        if not os.path.exists(f_path_path.parent.absolute()):
            os.makedirs(f_path_path.parent.absolute())

        with open(f_path, 'w') as f:
            f.write('CleanSlate')


def get_pid(out_command,out_file_path):
    
    make_retrieval_dt(out_file_path)
    runner_pid = start_proc(out_command,out_file_path)
    return runner_pid

def main():
    
    parser = ArgumentParser('This script is file watcher plus runner for Run.py of Ash Dieback project')
    
    parser.add_argument('--weights1', '-w1', help ='Path to yolov5 weights', 
                        required = False, default = 'weights/yolov5/best_1280.pt' )
    parser.add_argument('--weights2', '-w2', help ='Path to yolov5 weights', 
                        required = False, default = 'weights/yolov5/best_1280.pt' )
    parser.add_argument('--svo', '-s', help='Path to input svo folder', 
                        required=True)
    parser.add_argument('--resolution', '-r', help='resolution of the video', 
                        required=False, default='2k', type = str)
    parser.add_argument('--rear_videos', action="store_true")
    args = parser.parse_args()
    
    weights1 = args.weights1
    weights2 = args.weights2
    svo_location = args.svo
    resolution = args.resolution


    out_file_path = os.path.dirname(svo_location)
    out_file_path = os.path.join(svo_location, 'state.data')
    
    if args.rear_videos:
        out_command = ['python', 'Run.py', '--weights', weights1, weights2, 
                       '--svo', svo_location, '--resolution', resolution,'--rear_videos']
    
    else:
        out_command = ['python', 'Run.py', '--weights', weights1, weights2, 
                       '--svo', svo_location, '--resolution', resolution]
    
    print(' >>>>>>>>>>>>>>>>>>>>>> '.join(out_command))

    max_threshold = 30

    cur_pid = get_pid(out_command,out_file_path)
    print('Started Process with PID: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ', cur_pid)
    cur_state = 'CleanSlate'
    last_state_change_ts = time.time()
    
    while(True):
        if os.path.isfile(out_file_path):
            retrieval_state = get_retrieval_state(out_file_path)
        else:
            make_retrieval_dt(out_file_path)
            cur_state = 'CleanSlate'
            last_state_change_ts = time.time()

        # Check for CleanSlate.
        if retrieval_state == 'CleanSlate':
            # ignore, continue.
            if not cur_state == 'CleanSlate':
                last_state_change_ts = time.time()
            cur_state = retrieval_state
        
        elif retrieval_state == 'RetrievalStarted':
            
            if cur_state == 'CleanSlate':
                # If current state is this, ignore and carry on.
                cur_state = retrieval_state
                last_state_change_ts = time.time()
                
            elif cur_state == 'RetrievalStarted':
                # If time since last change is more than 10 seconds and a retrieval file is still in RetrievalState
                # - kill current process
                # - start a new process.
                # - reset retrieval state file.
                # - reset state variables to new.
                cur_ts = time.time()
                if cur_ts - last_state_change_ts > max_threshold:
                    kill_proc(cur_pid)
                    time.sleep(20)
                    print('Restarting Processing.')
                    
                    make_retrieval_dt(out_file_path)
                    cur_pid = main()
                    cur_state = 'CleanSlate'
                    last_state_change_ts = cur_ts
                    print('Retrieval running since: ', cur_ts - last_state_change_ts, ' duration.')
                    print('Retrieval active for more than ', max_threshold, ' duration.')
            
        elif retrieval_state == 'RetrievalComplete':

            cur_ts = time.time()
            # cur_pid = main()
            cur_state = 'CleanSlate'
            last_state_change_ts = cur_ts

        elif retrieval_state=="ProcessFinished":
            break
        
        else:
            # cur_pid = make_retrieval_dt(out_file_path)
            cur_state = 'CleanSlate'
            last_state_change_ts = cur_ts
    
        time.sleep(0.05)
    
    print("Watcher exited!")
    time.sleep(2)
    if os.path.isfile(out_file_path):
        os.remove(out_file_path)
    
if __name__ == '__main__': main()