#!/bin/bash
service ssh start
echo '[TARGET] SSH baslatildi'
service apache2 start
echo '[TARGET] Apache baslatildi'
tail -f /dev/null
