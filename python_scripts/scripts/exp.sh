#!/usr/bin/env bash
if [ ! -d experiments ];then
	mkdir experiments
fi

pushd experiments
	for seed in {0..10000..10000}
	do
		for env in Free Box Harsh
		do 
			for mapsize in 50. 100. 200. 
			do 
				for grad_step in 0.0 0.05 0.1 0.15 0.20
				do 
					for pathset in dubins
					do
						for nonmyopic in nonmyopic
						do	
							for reward_func in mean info_gain
							do
								echo sim_seed ${seed}-env ${env}-mapsize ${mapsize}-pathset ${pathset}-planner ${nonmyopic}-reward ${reward_func}

								if [ -d $workdir ] && [ -f ${workdir}/figures/${reward_func}/trajectory-N.SUMMARY.png ] ; then
									continue
								fi

								if [ -d $workdir ] && [ -f ${workdir}/figures/${reward_func}/trajectory-N.SUMMARY.png ] ; then
									continue
								fi

								# if [ ! -d $workdir ]; then mkdir $workdir; fi

								# pushd $workdir
								if [ ${nonmyopic} = nonmyopic ]; then
									cmd="python ../ipp_experiment.py -s ${seed} -r ${reward_func}
										-p ${pathset} -n ${nonmyopic} -e ${env} -z ${mapsize} -d ${grad_step}"
								fi

								echo $cmd
								$cmd
								# popd 
							done
						done
					done
				done
			done
		done 
	done
popd