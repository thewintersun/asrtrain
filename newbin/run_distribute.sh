#!/bin/bash

if [ $# != 1 ];then
   echo $0" run"
   echo $0" kill"
   echo $0" status"
   echo $0" sendfile"
   exit 0
fi

run_dir="/home/luodongri/code/git/asr/asr/train/bin/"

deploy_dir="/home/luodongri/code/git/asr/asr/train/bin1 /home/luodongri/code/git/asr/asr/train/bin2 /home/luodongri/code/git/asr/asr/train/bin3 /home/luodongri/code/git/asr/asr/train/bin4 /home/luodongri/code/git/asr/asr/train/bin5 /home/luodongri/code/git/asr/asr/train/bin6 /home/luodongri/code/git/asr/asr/train/bin7 /home/luodongri/code/git/asr/asr/train/bin8"
deploy_dir_list=(${deploy_dir// / })

ps_host="192.168.100.72:2200"

#格式是ip:port:python:第几个deploy_dir:第几个gpu
host72="192.168.100.72:2210:3:0:0,192.168.100.72:2211:3:1:0,192.168.100.72:2212:3:2:1,192.168.100.72:2213:3:3:1,192.168.100.72:2214:3:4:2,192.168.100.72:2215:3:5:2,192.168.100.72:2216:3:6:3,192.168.100.72:2217:3:7:3"
host73="192.168.100.73:2300:3:0:0,192.168.100.73:2301:3:1:0,192.168.100.73:2302:3:2:1,192.168.100.73:2303:3:3:1,192.168.100.73:2304:3:4:2,192.168.100.73:2305:3:5:2,192.168.100.73:2306:3:6:3,192.168.100.73:2307:3:7:3"
host74="192.168.100.74:2500:3:0:0,192.168.100.74:2501:3:1:0,192.168.100.74:2502:3:2:1,192.168.100.74:2503:3:3:1,192.168.100.74:2504:3:4:2,192.168.100.74:2505:3:5:2,192.168.100.74:2506:3:6:3,192.168.100.74:2507:3:7:3"
host75="192.168.100.75:2600:3:0:0,192.168.100.75:2601:3:1:0,192.168.100.75:2602:3:2:1,192.168.100.75:2603:3:3:1,192.168.100.75:2604:3:4:2,192.168.100.75:2605:3:5:2,192.168.100.75:2606:3:6:3,192.168.100.75:2607:3:7:3"
host76="192.168.100.76:2700:3:0:0,192.168.100.76:2701:3:1:0,192.168.100.76:2702:3:2:1,192.168.100.76:2703:3:3:1,192.168.100.76:2704:3:4:2,192.168.100.76:2705:3:5:2,192.168.100.76:2706:3:6:3,192.168.100.76:2707:3:7:3"

worker_host_all=$host72","$host73","$host74","$host75","$host76
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
    if [ "$ip" == "192.168.100.72" ];then
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
	ssh 192.168.100.72 "cd "${deploy_dir_list[0]}";source /etc/profile; python3 asr_distribute.py --cuda_visible_devices='' "$host_info" --job_name=ps --task_index=0 >nohup 2>&1 &"
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

	    ssh $ip "cd "${deploy_dir_list[$deploy_dir_index]}";source /etc/profile; python3 asr_distribute.py --cuda_visible_devices="$gpu_index" "$host_info" --job_name=worker --initial_learning_rate=0.0001 --train_feature_file=split_dir_"$deploy_dir_index"/train.feature --train_feature_len_file=split_dir_"$deploy_dir_index"/train.feature_len --train_label_file=split_dir_"$deploy_dir_index"/train.label --train_label_len_file=split_dir_"$deploy_dir_index"/train.label_len --task_index="$task_index" >nohup2 2>&1 &"
		else

	    ssh $ip "cd "${deploy_dir_list[$deploy_dir_index]}";source /etc/profile; python asr_distribute.py --cuda_visible_devices="$gpu_index" "$host_info" --job_name=worker --initial_learning_rate=0.0001  --train_feature_file=split_dir_"$deploy_dir_index"/train.feature --train_feature_len_file=split_dir_"$deploy_dir_index"/train.feature_len --train_label_file=split_dir_"$deploy_dir_index"/train.label --train_label_len_file=split_dir_"$deploy_dir_index"/train.label_len --task_index="$task_index" >nohup2 2>&1 &"
		fi
		i=$(expr $i + 1)

	done

echo "execute worker server command over"
fi
