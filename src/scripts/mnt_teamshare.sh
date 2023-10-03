# !/bin/bash
#mount.cifs //files.ubc.ca/team//BNRC/NINC /mnt -o credentials=/workspaces/Arjun_data/teamshare_credentials,uid=1000,gid=1000,sec=ntlmssp

docker volume create \
	--driver local \
	--opt type=cifs \
	--opt device=//files.ubc.ca/team//BNRC/NINC \
	--opt o=addr=files.ubc.ca,username=jtsuchit,password=Swalotpi1!,file_mode=0777,dir_mode=0777 \
	--name cif-volume
