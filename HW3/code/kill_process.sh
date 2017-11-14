#!/binbash 

echo "************** Killing Process ***************"

for pid in `ps -ux | grep python |grep utils.py|grep -v 'grep'| awk '{print $2}'`
do
  echo "----- kill $pid ****"  
  kill -9 $pid 
done 
echo "**************** Finish ********************" 
