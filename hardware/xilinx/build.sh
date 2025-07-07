#!/bin/sh

if false
then
	xsetup -b ConfigGen
	xsetup --agree XilinxEULA,3rdPartyEULA ,WebTalk --batch Install --config ~/.Xilinx/install_config.txt
	mkdir -p ~/bin
	ln -s /usr/bin/python3 ~/bin/python
	sudo apt install build-essentials libc6-dev-i386 zip
	echo Apply this patch 'https://adaptivesupport.amd.com/s/article/76960'
	echo Make sure you are on the following PYNQ release https://github.com/Xilinx/PYNQ/releases/tag/v2.5
fi

lsb_release -rs | grep -q '^18\.' || echo Warning you are not running on Ubuntu 18

cp -f ../../config/zcu104_sample.json ../../config/vta_conf.json
rm -rf ../../build
# https://adaptivesupport.amd.com/s/question/0D52E00006hpJpSSAU/
LIBRARY_PATH=/usr/lib/x86_64-linux-gnu make cleanall
LIBRARY_PATH=/usr/lib/x86_64-linux-gnu make
