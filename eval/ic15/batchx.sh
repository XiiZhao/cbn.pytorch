echo $1
for file in `cat $1`; do
    echo $file
    python script.py -g=gt.zip -s=$file
done    
