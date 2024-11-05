#!/bin/bash

# Download and install newer SQLite
wget https://www.sqlite.org/2024/sqlite-autoconf-3450100.tar.gz
tar xzf sqlite-autoconf-3450100.tar.gz
cd sqlite-autoconf-3450100
./configure
make
make install

# Update library path
echo "/usr/local/lib" > /etc/ld.so.conf.d/sqlite3.conf
ldconfig

# Clean up
cd ..
rm -rf sqlite-autoconf-3450100*
