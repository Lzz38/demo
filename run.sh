#! /bin/bash

#source flask/bin/activate
/home/security/.virtualenvs/sec/bin/gunicorn -w 32 -b unix:/tmp/skylab.sock --reload app:app
