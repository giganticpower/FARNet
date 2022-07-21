cd dataset/MLT2017/eval/
rm submit/*
cp $1/*.txt submit
cd submit/;zip -r  submit.zip * &> ../log.txt ;mv submit.zip ../; cd ../
rm log.txt
python2 script.py -g=mltGT.zip -s=submit.zip
