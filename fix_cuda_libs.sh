#!/bin/bash

# Use ldconfig to find the directory of libcuda.so
driver_dir=$(ldconfig -p | grep libcuda.so.1 | awk '{print $NF}' | xargs -I {} dirname {} | uniq)

echo $driver_dir
if [ -z "$driver_dir" ]; then
  echo "could not find driver."
  return 1
fi


# Loop through all *.so.1 files in the specified directory
for lib in "$driver_dir"/*.so.1; do
    # Check if the file exists to avoid errors
    if [ -e "$lib" ]; then
        # Get the base name without the .1
        base_name="${lib%.1}"

        # Check if the symlink (*.so) already exists
        if [ ! -e "$base_name" ]; then
            # Create the symbolic link
            ln -s "$lib" "$base_name"
            echo "Created symlink: $base_name -> $lib"
        else
            echo "Symlink already exists: $base_name"
        fi
    fi
done