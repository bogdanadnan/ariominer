#!/usr/bin/env bash

. h-manifest.conf

CUSTOM_API_PORT=`./parse-api-port.sh`

#try to release TIME_WAIT sockets
while true; do
	for con in `netstat -anp | grep TIME_WAIT | grep ${CUSTOM_API_PORT} | awk '{print $5}'`; do
		killcx $con lo
	done
	netstat -anp | grep TIME_WAIT | grep ${CUSTOM_API_PORT} &&
		continue ||
		break
done

echo -e "Running ${CYAN}ariominer${NOCOLOR}" | tee ${CUSTOM_LOG_BASENAME}.log

./ariominer --mode miner $(< $CUSTOM_NAME.conf)$@ 2>&1 | tee ${CUSTOM_LOG_BASENAME}.log
