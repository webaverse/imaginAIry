#!/bin/bash
source venv/bin/activate
sudo nohup $(which python) app.py >out.log 2>out.err < /dev/null &