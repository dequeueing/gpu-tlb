import os
import subprocess

# sudo = subprocess.Popen(['sudo', 'echo', 'What is your problem?'])
# sudo = subprocess.Popen(['sudo', 'echo', 'What is your problem?'])
sudo = subprocess.Popen('sudo echo What is your problem?', shell=True)
sudo.wait()
print("finished sudo")