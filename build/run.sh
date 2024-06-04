#!/bin/bash

# Preserve environment variables (slurm and some nvidia variables are set at runtime)
env | grep '^SLURM_\|^NVIDIA_' >> /etc/environment

# Disable python buffer for commands that are executed as user "user"
echo "PYTHONUNBUFFERED=1" >> /etc/environment

# Switch to codebase (defaults to home directory)
if [ -z "$CODEBASE" ] || ! cd "$CODEBASE"; then
  cd /home/user
else
  echo "CODEBASE=$CODEBASE" >> /etc/environment
fi
printf "Working directory: %s\n" "$(pwd)"

# Check if extra arguments were given and execute it as a command.
if [ -z "$2" ]; then
  # Print the command for logging.
  printf "No extra arguments given, starting sshd\n\n"

  # Start the SSH daemon and a Jupyter notebook.
  /usr/sbin/sshd
  sudo bash
else
  # Print the command for logging.
  printf "Executing command: %s\n\n" "$*"

  # Execute the passed command.
  sudo --user=user --set-home "${@}"
fi
