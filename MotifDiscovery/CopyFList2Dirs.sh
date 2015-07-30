#!/bin/bash
files=()

while read -r -d $'\0'; do
    files+=("$REPLY")
done < <(find $1 -iname "*.2s25motif" -print0)


for (( i = 0 ; i  < ${#files[@]}; i++ ))
do
        #echo ${files[$i]%.*}
        cp /home/sankalp/filelist.txt  ${files[$i]%.*}.flist
done