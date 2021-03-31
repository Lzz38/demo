#!/bin/bash
python3 manage.py delete_alembic
rm migrations -rf
python3 manage.py db init
python3 manage.py db migrate
python3 manage.py db upgrade
