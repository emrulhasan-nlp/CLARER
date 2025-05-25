#!/bin/bash
#nohup python codes/NRTPlus2.py > output1.log 2>&1 &
nohup python codes/NRTPlus2.py > output4.log 2>&1 &
nohup python codes/NRTPlus3.py > output5.log 2>&1 &
echo "Jobs submitted!"

