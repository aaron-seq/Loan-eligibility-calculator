from validators import LoanFormValidator, ValidationError

# Patch for app.py: server-side validation in /api/predict
# This file exists to guide reviewers; actual integration occurs by importing LoanFormValidator in app.py
