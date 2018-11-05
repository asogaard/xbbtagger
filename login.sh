#!/bin/bash
# Login to an Eddie3 compute node

function print_help () {
	echo "Bash script to automate login to Eddie3 compute nodes."
	echo ""
	echo "Arguments:"
	echo "  Parallel environment: ['sharedmem', 'gpu']."
	echo "      - Default: 'sharedmem'"
	echo "  Memory: Integer followed by g, gb, G, or GB, for instance '20GB'."
	echo "      - Default '10GB'"
	echo ""
	echo "Example commands:"
	echo "  source login.sh"
	echo "  source login.sh sharedmem 10g"
	echo "  source login.sh gpu 20GB"
}

# Extract flags
arguments=("$@")
set -- # Unsetting positional arguments, to avoid error from "source deactivate"

pe="sharedmem"
memory="10"
for arg in "${arguments[@]}"; do
    if   [[ "$arg" =~ .*elp$|.*-h.*$ ]]; then
	print_help
	return 0
    elif [ "$arg" == "sharedmem" ];  then
        pe="sharedmem"
    elif [ "$arg" == "gpu" ];  then
        pe="gpu"
    elif [[ "$arg" =~ ^[0-9]+[gG]+[bB]*$ ]]; then
        memory="$(sed 's/\([0-9]*\).*/\1/g' <<< $arg)"
    else
        echo "[ERROR] Argument '$arg' was not understood."
	return 1
    fi
done

# Compute number of nodes
if (( "$memory" > 40 )); then
    echo "[ERROR] Requested memory (${memory} GB) is excessive."
    return 1
elif (( "$memory" <= 0 )); then
    echo "[ERROR] Please specify a greater memory request than ${memory} GB."
    return 
fi
memory_per_node=10
nodes=$(( ($memory + $memory_per_node - 1) / $memory_per_node ))
# ^ Round up, such that we have at least `memory` available

function spinner {
    i=1
    sp="/-\|"
    echo -n ' '
    while true; do
	printf "\b\b\b[${sp:i++%${#sp}:1}]"
	sleep 0.1
    done
}

# Start login loop
counter=1
response="1"
logfile=".qlogin_output"
cmd="qlogin -pe ${pe} ${nodes} -l h_vmem=${memory_per_node}G" 
echo "Logging onto ${nodes} '${pe}' Eddie node(s) with ${memory_per_node} GB memory each."
echo "> ${cmd}"
while (( "$response" == 1 )); do
    # Logginng
    echo -n "[${counter}] Requesting login.   "
    start=`date +%s`
    # Start spinner
    spinner &
    pid_spinner=$!

    # Request login
    eval "${cmd} > $logfile 2>&1"
    response=$?
    
    # Silently kill spinner
    echo ""
    kill $pid_spinner
    wait $pid_spinner 2>/dev/null

    # Check response
    end=`date +%s`
    if (( "$(grep "Request for interactive job has been canceled." .qlogin_output | wc -l)" > 0 )); then
	echo "[ctrl+c] Exiting gracefully."
	response=4
    else
	echo "[${counter}] ... Attempt unsuccessful ($((end - start)) secs)."
	counter=$(($counter + 1))
    fi

    # Clean-up
    if [[ -f $logfile ]]; then
	rm $logfile
    fi
done

