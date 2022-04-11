sudo apt install net-tools -y
sudo apt install ifupdown
sudo service network-manager stop
# BaseStation Adapter Jetson Nano
echo "auto eth0
iface eth0 inet static
address 199.0.1.2
netmask 255.255.0.0
broadcast 199.199.0.0" | sudo tee --append /etc/network/interfaces
sudo service network-manager start
sudo ifup -a
