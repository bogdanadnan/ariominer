#!/usr/bin/env bash

. $MINER_DIR/$CUSTOM_MINER/h-manifest.conf

. $MINER_DIR/$CUSTOM_MINER/parse-api-port.sh

json_data=`curl -s http://localhost:${CUSTOM_API_PORT}/status`

cblocks_hashrates=`echo $json_data | jq -c ".hashers[]" | jq -c ".devices[]" | jq -c ".cblocks_hashrate"`
gblocks_hashrates=`echo $json_data | jq -c ".hashers[]" | jq -c ".devices[]" | jq -c ".gblocks_hashrate"`
block_height=`echo $json_data | jq -c ".block_height"`
cblocks_shares=`echo $json_data | jq -c ".cblocks_shares"`
gblocks_shares=`echo $json_data | jq -c ".gblocks_shares"`
cblocks_rejects=`echo $json_data | jq -c ".cblocks_rejects"`
gblocks_rejects=`echo $json_data | jq -c ".gblocks_rejects"`
uptime=`echo $json_data | jq -c ".time_running"`

total_cblocks_hashrate=`echo $cblocks_hashrates | awk '{sum=0; for(i=1; i<=NF; i++) sum += $i; sum = sum/1000; print sum}'`
total_gblocks_hashrate=`echo $gblocks_hashrates | awk '{sum=0; for(i=1; i<=NF; i++) sum += $i; sum = sum/1000; print sum}'`
total_shares=$((cblocks_shares+gblocks_shares))
total_rejects=$((cblocks_rejects+gblocks_rejects))

if (( block_height % 2 )); then
	khs=$total_gblocks_hashrate
	hashrates=$gblocks_hashrates
else
	khs=$total_cblocks_hashrate
	hashrates=$cblocks_hashrates
fi

stats=$(jq -nc \
	--argjson hs "`echo "$hashrates" | tr " " "\n" | jq -cs '.'`" \
	--arg hs_units "hs" \
	--arg uptime "$uptime" \
	--arg ac "$total_shares" --arg rj "$total_rejects" \
	--arg algo "argon2i" \
	'{$hs, $hs_units, $uptime,ar: [$ac, $rj], $algo}')
