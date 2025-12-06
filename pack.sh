#!/bin/bash

rm cirrus.zip
cd .. && zip -r cirrus.zip ./cirrus  -x "./cirrus/Cargo.lock" "./cirrus/target/*" "./cirrus/out/*" "./cirrus/key/*" "./cirrus/accountability_out/*" "./cirrus/.github/*" "./cirrus/.git/*" && mv cirrus.zip ./cirrus