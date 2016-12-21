#!/bin/bash

if [ $# != 1 ];then
   echo $0" run"
   echo $0" kill"
   echo $0" status"
   echo $0" sendfile"
   exit 0
fi

run_dir="/home/luodongri/code/git/asr/src/train/bin/"

deploy_dir="/home/luodongri/code/git/asr/src/train/bin /home/luodongri/code/git/asr/src/train/bin2 /home/luodongri/code/git/asr/src/train/bin3 /home/luodongri/code/git/asr/src/train/bin4 /home/luodongri/code/git/asr/src/train/bin5 /home/luodongri/code/git/asr/src/train/bin6"
deploy_dir_list=(${deploy_dir// / })

ps_host="192.168.100.62:2200"

#格式是ip:port:python:第几个deploy_dir:第几个gpu
host41="192.168.100.41:2210:3:0:0,192.168.100.41:2211:3:1:0,192.168.100.41:2212:3:2:1,192.168.100.41:2213:3:3:1,192.168.100.41:2214:3:4:2,192.168.100.41:2215:3:5:2"
host42="192.168.100.42:2300:3:0:0,192.168.100.42:2301:3:1:0,192.168.100.42:2302:3:2:1,192.168.100.42:2303:3:3:1"
host65="192.168.100.65:2500:2:0:2,192.168.100.65:2501:2:1:2,192.168.100.65:2502:2:2:3,192.168.100.65:2503:2:3:3"
host62="192.168.100.62:2600:3:0:0,192.168.100.62:2601:3:1:1,192.168.100.62:2602:3:2:2"
host22="192.168.100.22:2700:2:0:0"
host253="192.168.100.253:2800:3:0:0"
host30="192.168.100.30:2900:3:0:0"

worker_host_all=$host62","$host30
worker_host_all_list=${worker_host_all//,/ }

#提取出来ip:port
worker_host=""
for host in $worker_host_all_list; do
	host_array=(${host//:/ })
	ip=${host_array[0]}
	port=${host_array[1]}
	if [ "$worker_host" == "" ];then
		worker_host=$ip":"$port
	else
		worker_host=$worker_host","$ip":"$port
	fi
done
echo $worker_host

host_info="--ps_hosts="$ps_host" --worker_hosts="$worker_host


#ssh到每台机器上， kill到进程
if [ "$1" == "kill" ];then
   array_str=${worker_host//,/ }
   for host in $array_str; do
    host_array=(${host//:/ })
    ip=${host_array[0]}
    ssh $ip "ps aux | grep python | grep asr_distribute.py | awk '{print \$2;}' | xargs kill -9 "
	sleep 1s
    done
	exit 0
fi


# ssh到每台机器上， 查看有几个进程还在运行，不过第一个机器正常情况下会多2个, 一个是ps进程，一个是执行命令本身造成的
if [ "$1" == "status" ];then
  array_str=${worker_host//,/ }
  for host in $array_str; do
    host_array=(${host//:/ })
    ip=${host_array[0]}
    wcl=$(ssh $ip "ps aux | grep python | grep asr_distribute.py "  | wc -l)
    processnum=$(expr $wcl - 1)
    echo $ip" "$processnum
    sleep 1s
  done
  exit 0
fi

if [ "$1" == "sendfile" ];then
  #自己写命令到这里
  array_str=${worker_host//,/ }
  for host in $array_str; do
    host_array=(${host//:/ })
    ip=${host_array[0]}
  	echo "send to "$ip
    if [ "$ip" == "192.168.100.62" ];then
      echo "jump "$ip
    else
      #scp $run_dir"/asr.py" $ip":"$run_dir
	    scp $run_dir"/asr_distribute.py" $ip":"$run_dir
      
    fi

done
  
fi



#开始执行命令
if [ "$1" == "run" ];then
	#先ssh到ps服务器， 执行ps命令
	echo "execute ps server command"
	ssh 192.168.100.62 "cd "${deploy_dir_list[0]}";source /etc/profile; python3 asr_distribute.py --cuda_visible_devices='' "$host_info" --job_name=ps --task_index=0 >nohup 2>&1 &"
	sleep 1s
	echo "execute ps server command over"

	#执行worker节点
	echo "execute worker server command"
	worker_host_str=${worker_host_all//,/ }
	worker_host_list=(${worker_host_all//,/ })
	host_num=${#worker_host_list[@]}
	echo "host number is "$host_num
	
	i=0
	for host in $worker_host_str; do
		host_array=(${host//:/ })
		ip=${host_array[0]}
		port=${host_array[1]}
		pythonflag=${host_array[2]}
		deploy_dir_index=${host_array[3]}
		gpu_index=${host_array[4]}
		task_index=$i
		
		echo $task_index" "$ip
		
		if [ $pythonflag -eq "3" ];then

			ssh $ip "cd "${deploy_dir_list[$deploy_dir_index]}";source /etc/profile; python3 asr_distribute.py --cuda_visible_devices="$gpu_index" "$host_info" --job_name=worker --initial_learning_rate=0.0001 --task_index="$task_index" >nohup2 2>&1 &"
		else

			ssh $ip "cd "${deploy_dir_list[$deploy_dir_index]}";source /etc/profile; python asr_distribute.py --cuda_visible_devices="$gpu_index" "$host_info" --job_name=worker --initial_learning_rate=0.0001 --task_index="$task_index" >nohup2 2>&1 &"
		fi
		i=$(expr $i + 1)

	done

echo "execute worker server command over"
fi


