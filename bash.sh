#!/bin/bash

srun -Q --immediate=10 --partition=all_serial --gres=gpu:1 --account=cvcs2024 --time 60:00 --pty bash
